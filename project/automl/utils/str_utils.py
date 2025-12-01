
def generate_unique_versioned_str(base_str : str, current_collection : str):

    if base_str in current_collection:

        i = 1
        str_to_return = f"{base_str}_{i}"

        while str_to_return in current_collection:
            i += 1
            str_to_return = f"{base_str}_{i}"

        return str_to_return
    
    else:
        return base_str