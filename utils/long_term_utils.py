import torch
import torch.nn.functional as F
import signal
from kmeans_pytorch import kmeans


class TimeoutException(Exception):
    """Exception raised when operation times out."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException()


def _k_means_with_timeout(x, n_mem, timeout_seconds=1):

    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

        _, cluster_centers = kmeans(
            X=x,
            num_clusters=n_mem,
            distance='euclidean',
            device=x.device,
            tol=1e-3,
            tqdm_flag=False
        )

        signal.alarm(0)
        return cluster_centers, True

    except (TimeoutException, Exception):
        signal.alarm(0)
        return None, False


def _random_sampling_fallback(x, n_mem):

    device = x.device
    dtype = x.dtype

    unique_data = torch.unique(x, dim=0)
    n_unique = unique_data.shape[0]

    if n_unique >= n_mem:
        indices = torch.randperm(n_unique, device=device)[:n_mem]
        centers = unique_data[indices]
    else:
        centers_from_data = unique_data
        additional_needed = n_mem - n_unique
        random_centers = torch.randn(additional_needed, x.shape[1], device=device, dtype=dtype)
        centers = torch.cat([centers_from_data, random_centers], dim=0)

    return centers.to(device=device, dtype=dtype)


def _mean_initialization(x):

    device = x.device
    dtype = x.dtype

    center = x.mean(dim=0, keepdim=True)
    center = F.normalize(center, dim=-1)

    return center.to(device=device, dtype=dtype)


def _diagnose_input(x):

    nan_count = torch.isnan(x).sum().item()
    inf_count = torch.isinf(x).sum().item()

    norms = torch.norm(x, dim=1)
    zero_norm_count = (norms < 1e-8).sum().item()

    return nan_count == 0 and inf_count == 0 and zero_norm_count == 0


def long_term_k_means_clustering(x, n_mem, d_model, timeout_seconds=1):

    _diagnose_input(x)
    x = x.view([-1, d_model])

    if n_mem == 1:
        return _mean_initialization(x)

    cluster_centers, success = _k_means_with_timeout(x, n_mem, timeout_seconds)

    if not success:
        cluster_centers = _random_sampling_fallback(x, n_mem)

    return cluster_centers


def fixed_long_term_k_means_clustering(x, n_mem, d_model, timeout_per_dim=1):

    batch_size, n_dims, feature_dim = x.shape
    device = x.device
    dtype = x.dtype

    all_cluster_centers = []

    for dim_idx in range(n_dims):
        dim_data = x[:, dim_idx, :]

        if n_mem == 1:
            centers = _mean_initialization(dim_data)
            all_cluster_centers.append(centers.to(device=device, dtype=dtype))
            continue

        is_valid = _diagnose_input(dim_data)

        if not is_valid:
            centers = _random_sampling_fallback(dim_data, n_mem)
            all_cluster_centers.append(centers.to(device=device, dtype=dtype))
            continue

        unique_data = torch.unique(dim_data, dim=0)
        current_n_mem = min(n_mem, unique_data.shape[0])

        cluster_centers, success = _k_means_with_timeout(
            unique_data, current_n_mem, timeout_per_dim
        )

        if success:
            cluster_centers = cluster_centers.to(device=device, dtype=dtype)
            if cluster_centers.shape[0] < n_mem:
                extra = torch.randn(
                    n_mem - cluster_centers.shape[0],
                    feature_dim,
                    device=device,
                    dtype=dtype
                )
                cluster_centers = torch.cat([cluster_centers, extra], dim=0)
            all_cluster_centers.append(cluster_centers)
        else:
            centers = _random_sampling_fallback(dim_data, n_mem)
            all_cluster_centers.append(centers.to(device=device, dtype=dtype))

    return torch.stack(all_cluster_centers, dim=0)