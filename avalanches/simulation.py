from avalanches.models import MLPWithPerturbationStop
import torch, numpy as np, os

def run_simulation_manybody(N, L_max, sigma_w2, sigma_b2, num_samples=1000,
                          print_every=1000, q0 = 0.01, no_shape = True):

    os.makedirs("results", exist_ok=True)

    model = MLPWithPerturbationStop(N, L_max, sigma_w2, sigma_b2)
    recording = []
    for i in range(1, num_samples + 1):
        if i % 1000 ==0:
            model = MLPWithPerturbationStop(N, L_max, sigma_w2, sigma_b2)
        x = torch.empty(N).normal_(mean=0, std=1)
        x = (x / torch.norm(x)) * np.sqrt(q0)
        _, avalanche = model(x, threshold=q0)
        if len(avalanche) > 2: #Check if there was an avalanche at all
            avalanche = avalanche[1:-2]- np.sqrt(q0)
            if no_shape:
                # (size , duration) tupple
                recording.append(   (np.sum(avalanche),len(avalanche))   )
            else :
                # full avalanche profile
                recording.append(avalanche)

        if print_every and i % print_every == 0:
            print(f"[N={N}, L={L_max}, sigma_w2={sigma_w2}] {i}/{num_samples} ({(i/num_samples)*100:.2f}%) done")

    return recording