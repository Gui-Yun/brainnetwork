def compute_components_discrete(X_int):
    """
    X_int: (N, k) discrete ints
    returns dict of components
    """
    k = X_int.shape[1]
    H_joint = joint_entropy_discrete(X_int)

    H_single_sum = 0.0
    for i in range(k):
        H_single_sum += joint_entropy_discrete(X_int[:, i:i+1])

    TC = H_single_sum - H_joint
    Omega = o_information_discrete(X_int)
    DTC = TC - Omega  # since Omega = TC - DTC

    return {
        "H_joint": H_joint,
        "H_single_sum": H_single_sum,
        "H_single_mean": H_single_sum / k,
        "TC": TC,
        "DTC": DTC,
        "Omega": Omega,
    }
# ---------- Discretization ----------
def discretize_continuous(X, n_bins=3, strategy="quantile"):
    """
    X: (N, k) continuous
    strategy: "uniform" (recommended for comparing H/TC across conditions),
              or "quantile" (more robust but may wash out marginal differences)
    """
    # z-score per neuron (column), across trials
    Xz = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)

    if strategy == "quantile":
        disc = KBinsDiscretizer(
            n_bins=n_bins,
            encode="ordinal",
            strategy="quantile",
            quantile_method="averaged_inverted_cdf"
        )
    elif strategy == "uniform":
        disc = KBinsDiscretizer(
            n_bins=n_bins,
            encode="ordinal",
            strategy="uniform"
        )
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    return disc.fit_transform(Xz).astype(np.int64)

# ---------- Timecourse via neuron subsampling ----------
def timecourse_components_subsample(
    data, k=10, n_repeat=300, n_bins=3, strategy="uniform", seed=0
):
    """
    data: (N, n, t) continuous
    returns:
        out_mean: dict metric -> (t,)
        out_sem:  dict metric -> (t,)
    """
    rng = np.random.default_rng(seed)
    N, n, t = data.shape

    metrics = ["H_joint", "H_single_sum", "H_single_mean", "TC", "DTC", "Omega"]
    out_mean = {m: np.zeros(t) for m in metrics}
    out_sem  = {m: np.zeros(t) for m in metrics}

    for tt in range(t):
        samples = {m: [] for m in metrics}
        for _ in range(n_repeat):
            idx = rng.choice(n, k, replace=False)
            X = data[:, idx, tt]  # (N, k)

            X_int = discretize_continuous(X, n_bins=n_bins, strategy=strategy)
            comps = compute_components_discrete(X_int)

            for m in metrics:
                samples[m].append(comps[m])

        for m in metrics:
            arr = np.asarray(samples[m], dtype=float)
            out_mean[m][tt] = arr.mean()
            out_sem[m][tt]  = arr.std(ddof=1) / np.sqrt(len(arr))

    return out_mean, out_sem

def compute_per_class_balanced(
    segments_flo, labels_flo,
    time_slice=None,
    k=10, n_repeat=300, n_bins=3, strategy="quantile",
    seed=0
):
    rng = np.random.default_rng(seed)

    unique_classes, counts = np.unique(labels_flo, return_counts=True)
    m = counts.min()
    print("Class counts:", dict(zip(unique_classes, counts)))
    print("Using balanced n_per_class =", m)

    if time_slice is None:
        data_all = segments_flo
    else:
        data_all = segments_flo[:, :, time_slice]

    results = {}  # class_label -> (mean_dict, sem_dict)

    for class_label in unique_classes:
        idx = np.where(labels_flo == class_label)[0]
        idx_sub = rng.choice(idx, size=m, replace=False)

        class_data = data_all[idx_sub, :, :]  # (m, n, t)
        mean_dict, sem_dict = timecourse_components_subsample(
            class_data, k=k, n_repeat=n_repeat, n_bins=n_bins,
            strategy=strategy, seed=seed
        )
        results[class_label] = (mean_dict, sem_dict)

    return results, m


def plot_metric_timecourses(results, metric, stim_on=None, title=None):
    plt.figure(figsize=(8, 4.5))
    for class_label, (mean_dict, sem_dict) in results.items():
        mu = mean_dict[metric]
        se = sem_dict[metric]
        x = np.arange(len(mu))
        plt.plot(x, mu, label=f"class {class_label}")
        plt.fill_between(x, mu - se, mu + se, alpha=0.2)

    if stim_on is not None:
        plt.axvline(stim_on, linestyle="--")

    plt.xlabel("Time points")
    plt.ylabel(metric)
    plt.title(title if title is not None else f"{metric} timecourse (balanced trials)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_metric_bar_stimwindow(results, metric, stim_window_slice, title=None):
    labels, means, errs = [], [], []
    for class_label, (mean_dict, sem_dict) in results.items():
        mu = mean_dict[metric][stim_window_slice]
        se = sem_dict[metric][stim_window_slice]

        val = mu.mean()
        err = np.sqrt(np.mean(se**2))  # RMS 合成一个保守误差

        labels.append(f"class {class_label}")
        means.append(val)
        errs.append(err)

    x = np.arange(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar(x, means, yerr=errs, capsize=4)
    plt.xticks(x, labels)
    plt.ylabel(metric)
    plt.title(title if title is not None else f"{metric} (stim window)")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()
