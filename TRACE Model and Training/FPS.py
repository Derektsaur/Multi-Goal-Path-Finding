import time
import torch
# from NEED import*
from Models import*
# from GPM import*
from torchsummary import summary

def measure_inference_speed(model, data, max_iter=2000, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps

if __name__ == '__main__':
    net = Transformer(10, 8, 256, 128, 256, 512, None, 0.1, 16*16, [16, 16]).to(device='cuda')
    net = net.cuda()
    print(torch.cuda.is_available())
    data = torch.randn((1, 3, 256, 256)).cuda()
    # summary(net, (3, 256, 256))
    measure_inference_speed(net, (data,))