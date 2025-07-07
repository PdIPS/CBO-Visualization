viridis_data = [
    (68, 1, 84),
    (71, 44, 122),
    (59, 81, 139),
    (44, 113, 142),
    (33, 144, 141),
    (39, 173, 129),
    (92, 200, 99),
    (170, 220, 50),
    (253, 231, 37),
]

def viridis(t):
    # t ∈ [0,1], interpolate in viridis_data
    n = len(viridis_data)
    idx = t * (n - 1)
    i0 = int(idx)
    i1 = min(i0 + 1, n - 1)
    f = idx - i0

    c0 = viridis_data[i0]
    c1 = viridis_data[i1]

    r = int(c0[0] + f * (c1[0] - c0[0]))
    g = int(c0[1] + f * (c1[1] - c0[1]))
    b = int(c0[2] + f * (c1[2] - c0[2]))

    return f"rgb({r},{g},{b})"