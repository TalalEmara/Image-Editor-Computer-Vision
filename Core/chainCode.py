
import numpy as np

directions = {
    (1, 0): 0,  # Right
    (1, 1): 1,  # Bottom-right
    (0, 1): 2,  # Down
    (-1, 1): 3,  # Bottom-left
    (-1, 0): 4,  # Left
    (-1, -1): 5,  # Top-left
    (0, -1): 6,  # Up
    (1, -1): 7   # Top-right
}

def getChainCode(contour: np.ndarray):
    chainCode = []
    previousPoint = tuple(contour[0])

    for point in contour[1:]:
        point = tuple(point)
        difference = (int(point[0] - previousPoint[0]), int(point[1] - previousPoint[1]))

        chainCode.append(directions[difference])

        previousPoint = point

    chainCode = normalize_chain_code(chainCode)
    chainCode = differenceNorm(chainCode)

    return chainCode

def differenceNorm(chainCode):
    return [(chainCode[i] - chainCode[i - 1]) % 8 for i in range(1, len(chainCode))]

def normalize_chain_code(chain_code):
    min_value = min(chain_code)  # O(n)
    min_indices = [i for i, val in enumerate(chain_code) if val == min_value]  # O(n)
    rotations = [chain_code[i:] + chain_code[:i] for i in min_indices]  # O(k * n), where k is the number of min occurrences
    return min(rotations)  # O(k * n) in worst case




if __name__ == "__main__":

    import numpy as np
    import time


    # Assuming the Chain Code functions are defined as provided earlier

    def test_getChainCode():
        test_cases = [
            {
                "description": "Square contour",
                "contour": np.array([[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]),
                "expected_chain_code": [0, 2, 4, 6],
                "expected_normalized": [0, 2, 4, 6]
            },
            {
                "description": "Diagonal line",
                "contour": np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
                "expected_chain_code": [1, 1, 1],
                "expected_normalized": [1, 1, 1]
            },
            {
                "description": "Single point (edge case)",
                "contour": np.array([[0, 0]]),
                "expected_chain_code": [],
                "expected_normalized": []
            },
            {
                "description": "Collinear points (edge case)",
                "contour": np.array([[0, 0], [1, 0], [2, 0], [3, 0]]),
                "expected_chain_code": [0, 0, 0],
                "expected_normalized": [0, 0, 0]
            },
            {
                "description": "Complex shape (e.g., 'L' shape)",
                "contour": np.array([[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]),
                "expected_chain_code": [2, 2, 0, 0],
                "expected_normalized": [0, 0, 2, 2]  # Expected normalized chain
            }
        ]

        for i, test in enumerate(test_cases):
            contour = test["contour"]
            expected_chain = test["expected_chain_code"]
            expected_norm = test["expected_normalized"]
            try:
                start_time = time.time()
                result_chain = getChainCode(contour)
                result_norm = normalize_chain_code(result_chain)
                end_time = time.time()
                status_chain = "PASSED" if result_chain == expected_chain else "FAILED"
                status_norm = "PASSED" if result_norm == expected_norm else "FAILED"
            except Exception as e:
                result_chain = str(e)
                result_norm = str(e)
                status_chain = status_norm = "ERROR"

            print(f"Test Case {i + 1}: {test['description']}")
            print(f"  Chain Code Status: {status_chain}")
            print(f"  Normalized Chain Code Status: {status_norm}")
            print(f"  Input Contour: {contour.tolist()}")
            print(f"  Expected Chain Code: {expected_chain}")
            print(f"  Actual Chain Code: {result_chain}")
            print(f"  Expected Normalized Chain Code: {expected_norm}")
            print(f"  Actual Normalized Chain Code: {result_norm}")
            print(f"  Execution Time: {end_time - start_time:.6f} seconds\n")



    test_getChainCode()