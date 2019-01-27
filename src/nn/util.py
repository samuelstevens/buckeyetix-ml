def get_nodes(arr=[], max_length=4, min=3, max=10):
    arr_of_arrs = []

    if len(arr) >= max_length:
        return []

    arr.append(0)

    if len(arr) > 1:
        for i in range(min, arr[-2] + 1):
            arr[-1] = i
            arr_of_arrs.append(arr[:])
            arr_of_arrs = arr_of_arrs + get_nodes(arr[:], max_length)
    else:
        for i in range(min, max + 1):
            arr[-1] = i
            arr_of_arrs.append(arr[:])
            arr_of_arrs = arr_of_arrs + get_nodes(arr[:], max_length)

    return arr_of_arrs
