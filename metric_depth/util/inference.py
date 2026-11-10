import torch
import torch.nn.functional as F


def tile_starts(length, tile_size, step):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, step))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


@torch.no_grad()
def predict_height_tiled(model, image, tile_size=518, overlap=126, batch_size=1, *, return_mag_scale=False):
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError('Tiled inference expects one NCHW image')
    if tile_size < 14 or tile_size % 14:
        raise ValueError('tile_size must be a positive multiple of 14')
    if not 0 <= overlap < tile_size:
        raise ValueError('overlap must be between 0 and tile_size - 1')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    height, width = image.shape[-2:]
    pad_height = max(0, tile_size - height)
    pad_width = max(0, tile_size - width)
    if pad_height or pad_width:
        image = F.pad(image, (0, pad_width, 0, pad_height), mode='replicate')
    padded_height, padded_width = image.shape[-2:]
    ys = tile_starts(padded_height, tile_size, tile_size - overlap)
    xs = tile_starts(padded_width, tile_size, tile_size - overlap)
    taper = 0.05 + 0.95 * torch.hann_window(
        tile_size, periodic=False, device=image.device, dtype=torch.float32
    )
    weights = taper[:, None] * taper[None, :]
    channels = 2 if return_mag_scale else 1
    prediction_sum = torch.zeros((channels, padded_height, padded_width), device=image.device)
    weight_sum = torch.zeros_like(prediction_sum[0])
    if return_mag_scale:
        scale_sum = image.new_zeros((), dtype=torch.float32)
    positions = [(top, left) for top in ys for left in xs]
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start:start + batch_size]
        tiles = torch.cat([
            image[:, :, top:top + tile_size, left:left + tile_size]
            for top, left in batch_positions
        ])
        output = model(tiles)
        prediction = output['depth'] if isinstance(output, dict) else output
        if return_mag_scale:
            prediction = torch.stack((prediction, output['mag']), dim=1)
            scale = output['scale']
            if scale.shape != (len(batch_positions),) or not torch.isfinite(scale).all():
                raise ValueError('Invalid scale predictions during tiled inference')
            scale_sum += scale.float().sum()
        else:
            prediction = prediction[:, None]
        if prediction.shape != (len(batch_positions), channels, tile_size, tile_size):
            raise ValueError(f'Unexpected tile prediction shape: {prediction.shape}')
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Non-finite map prediction during tiled inference')
        for index, (top, left) in enumerate(batch_positions):
            prediction_sum[:, top:top + tile_size, left:left + tile_size] += prediction[index].float() * weights
            weight_sum[top:top + tile_size, left:left + tile_size] += weights
    result = (prediction_sum / weight_sum)[:, :height, :width]
    if return_mag_scale:
        return {'depth': result[:1], 'mag': result[1:], 'scale': scale_sum / len(positions)}
    return result
