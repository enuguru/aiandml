

def mean(data):
    """Calculate the arithmetic mean of a list of numbers.
    
    Example:
        >>> numbers = [1, 2, 3, 4, 5]
        >>> mean(numbers)
        3.0
    """
    return sum(data) / len(data)

def median(data):
    """Calculate the median of a list of numbers.
    
    Example:
        >>> numbers = [1, 3, 2, 5, 4]
        >>> median(numbers)
        3
        >>> numbers = [1, 2, 3, 4]
        >>> median(numbers)
        2.5
    """
    sorted_data = sorted(data)
    n = len(data)
    if n % 2 == 0:
        return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    else:
        return sorted_data[n // 2]

def mode(data):
    """Find the mode (most frequent value) in a list of numbers.
    
    Example:
        >>> numbers = [1, 2, 2, 3, 3, 3, 4]
        >>> mode(numbers)
        3
    """
    counts = {}
    for value in data:
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = 1
    return max(counts, key=counts.get)


if __name__ == "__main__":
    numbers = [1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 7]
    print(f"List of numbers: {numbers}")
    print(f"Mean: {mean(numbers)}")
    print(f"Median: {median(numbers)}")
    print(f"Mode: {mode(numbers)}")
