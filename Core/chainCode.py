
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
    """Compute the Freeman Chain Code from a given contour."""
    if len(contour) < 2:
        return []

    chainCode = []
    previousPoint = tuple(contour[0])

    for point in contour[1:]:
        point = tuple(point)
        difference = (point[0] - previousPoint[0], point[1] - previousPoint[1])
        if difference in directions:
            chainCode.append(directions[difference])
        previousPoint = point

    chainCode = normalize_chain_code(chainCode)
    return differenceNorm(chainCode)

def differenceNorm(chainCode):
    """Compute the first difference of the chain code."""
    return [(chainCode[i] - chainCode[i - 1]) % 8 for i in range(1, len(chainCode))]

def normalize_chain_code(chain_code):
    """Normalize the chain code to its lexicographically smallest rotation."""
    if not chain_code:
        return []

    min_value = min(chain_code)
    min_indices = [i for i, val in enumerate(chain_code) if val == min_value]
    rotations = [chain_code[i:] + chain_code[:i] for i in min_indices]
    return min(rotations)

def compute_perimeter(chain_code):
    """Compute the perimeter based on Freeman Chain Code."""
    step_lengths = {
        0: 1, 1: np.sqrt(2), 2: 1, 3: np.sqrt(2),
        4: 1, 5: np.sqrt(2), 6: 1, 7: np.sqrt(2)
    }
    return sum(step_lengths[code] for code in chain_code)

def compute_area(chain_code):
    """Compute the area enclosed by the chain code using the Shoelace formula."""
    moves = {
        0: (1, 0), 1: (1, -1), 2: (0, -1), 3: (-1, -1),
        4: (-1, 0), 5: (-1, 1), 6: (0, 1), 7: (1, 1)
    }

    # Reconstruct the contour assuming the start point is (0,0)
    x, y = 0, 0
    contour = [(x, y)]

    for code in chain_code:
        dx, dy = moves[code]
        x, y = x + dx, y + dy
        contour.append((x, y))

    # Compute the area using the Shoelace formula
    n = len(contour)
    area = 0
    for i in range(n - 1):
        x1, y1 = contour[i]
        x2, y2 = contour[i + 1]
        area += (x1 * y2 - x2 * y1)

    return abs(area) / 2


# if __name__ == "__main__":
#
#     import numpy as np
#     import time
#
#
#     # Assuming the Chain Code functions are defined as provided earlier
#
#     def test_getChainCode():
#         test_cases = [
#             {
#                 "description": "Square contour",
#                 "contour": np.array([[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]]),
#                 "expected_chain_code": [0, 2, 4, 6],
#                 "expected_normalized": [0, 2, 4, 6]
#             },
#             {
#                 "description": "Diagonal line",
#                 "contour": np.array([[0, 0], [1, 1], [2, 2], [3, 3]]),
#                 "expected_chain_code": [1, 1, 1],
#                 "expected_normalized": [1, 1, 1]
#             },
#             {
#                 "description": "Single point (edge case)",
#                 "contour": np.array([[0, 0]]),
#                 "expected_chain_code": [],
#                 "expected_normalized": []
#             },
#             {
#                 "description": "Collinear points (edge case)",
#                 "contour": np.array([[0, 0], [1, 0], [2, 0], [3, 0]]),
#                 "expected_chain_code": [0, 0, 0],
#                 "expected_normalized": [0, 0, 0]
#             },
#             {
#                 "description": "Complex shape (e.g., 'L' shape)",
#                 "contour": np.array([[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]),
#                 "expected_chain_code": [2, 2, 0, 0],
#                 "expected_normalized": [0, 0, 2, 2]  # Expected normalized chain
#             }
#         ]
#
#         for i, test in enumerate(test_cases):
#             contour = test["contour"]
#             expected_chain = test["expected_chain_code"]
#             expected_norm = test["expected_normalized"]
#             try:
#                 start_time = time.time()
#                 result_chain = getChainCode(contour)
#                 result_norm = normalize_chain_code(result_chain)
#                 end_time = time.time()
#                 status_chain = "PASSED" if result_chain == expected_chain else "FAILED"
#                 status_norm = "PASSED" if result_norm == expected_norm else "FAILED"
#             except Exception as e:
#                 result_chain = str(e)
#                 result_norm = str(e)
#                 status_chain = status_norm = "ERROR"
#
#             print(f"Test Case {i + 1}: {test['description']}")
#             print(f"  Chain Code Status: {status_chain}")
#             print(f"  Normalized Chain Code Status: {status_norm}")
#             print(f"  Input Contour: {contour.tolist()}")
#             print(f"  Expected Chain Code: {expected_chain}")
#             print(f"  Actual Chain Code: {result_chain}")
#             print(f"  Expected Normalized Chain Code: {expected_norm}")
#             print(f"  Actual Normalized Chain Code: {result_norm}")
#             print(f"  Execution Time: {end_time - start_time:.6f} seconds\n")



    test_getChainCode()

if __name__ == "__main__":

    import numpy as np

    def test_compute_perimeter():
        test_cases = [
            {
                "description": "Square (4 sides of length 1)",
                "chain_code": [0, 2, 4, 6],
                "expected_perimeter": 4
            },
            {
                "description": "Diagonal line (3 diagonal moves)",
                "chain_code": [1, 1, 1],
                "expected_perimeter": 3 * np.sqrt(2)
            },
            {
                "description": "L shape",
                "chain_code": [2, 2, 0, 0],
                "expected_perimeter": 2 + 2
            },
            {
                "description": "Zigzag shape",
                "chain_code": [1, 7, 1, 7],
                "expected_perimeter": 4 * np.sqrt(2)
            }
        ]

        for i, test in enumerate(test_cases):
            chain_code = test["chain_code"]
            expected_perimeter = test["expected_perimeter"]
            try:
                result_perimeter = compute_perimeter(chain_code)
                status = "PASSED" if np.isclose(result_perimeter, expected_perimeter, atol=1e-6) else "FAILED"
            except Exception as e:
                result_perimeter = str(e)
                status = "ERROR"

            print(f"Test Case {i + 1}: {test['description']}")
            print(f"  Perimeter Status: {status}")
            print(f"  Chain Code: {chain_code}")
            print(f"  Expected Perimeter: {expected_perimeter}")
            print(f"  Actual Perimeter: {result_perimeter}\n")


    import numpy as np


    def test_compute_area():
        test_cases = [
            {
                "description": "Square (2x2 units)",
                "chain_code": [0, 6, 6, 4, 4, 2, 2, 0],  # Corrected chain code
                "expected_area": 4
            },
            {
                "description": "Triangle (Right-angled, base=2, height=2)",
                "chain_code": [0, 0, 6, 6, 3, 3],  # Corrected chain code
                "expected_area": 2
            },
            {
                "description": "Rectangle (3x2 units)",
                "chain_code": [0, 0, 6, 6, 4, 4,4, 2, 2, 0, 0],  # Corrected chain code
                "expected_area": 6
            }
        ]

        for i, test in enumerate(test_cases):
            chain_code = test["chain_code"]
            expected_area = test["expected_area"]
            try:
                result_area = compute_area(chain_code)
                status = "PASSED" if np.isclose(result_area, expected_area, atol=1e-6) else "FAILED"
            except Exception as e:
                result_area = str(e)
                status = "ERROR"

            print(f"Test Case {i + 1}: {test['description']}")
            print(f"  Area Status: {status}")
            print(f"  Chain Code: {chain_code}")
            print(f"  Expected Area: {expected_area}")
            print(f"  Actual Area: {result_area}\n")



    print("Testing Perimeter Calculation...\n")
    test_compute_perimeter()

    print("\nTesting Area Calculation...\n")
    test_compute_area()
