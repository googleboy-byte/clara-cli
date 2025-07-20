def generate_representative_test_set(all_filenames, toi_filenames, sample_size, seed=42):
    import random
    """
    Returns a representative random sample of .fits files with the same TOI ratio as the full sector.

    Args:
        all_filenames (list): All light curve .fits filenames from the sector.
        toi_filenames (list): Subset of those which are TOIs.
        sample_size (int): Desired total number of samples in test set.
        seed (int): Random seed for reproducibility.

    Returns:
        list: List of .fits filenames for the representative test set.
    """
    random.seed(seed)

    toi_set = set(toi_filenames)
    all_set = set(all_filenames)

    # Make sure TOIs are actually in the available pool
    toi_files = [f for f in all_filenames if f in toi_set]
    non_toi_files = [f for f in all_filenames if f not in toi_set]

    # Compute ratio
    toi_ratio = len(toi_files) / len(all_filenames)
    n_toi = round(sample_size * toi_ratio)
    n_non_toi = sample_size - n_toi

    # Sample
    sampled_tois = random.sample(toi_files, min(n_toi, len(toi_files)))
    sampled_non_tois = random.sample(non_toi_files, n_non_toi)

    # Combine and shuffle
    final_sample = sampled_tois + sampled_non_tois
    random.shuffle(final_sample)

    return final_sample