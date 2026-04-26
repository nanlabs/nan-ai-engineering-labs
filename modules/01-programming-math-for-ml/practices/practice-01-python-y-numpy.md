# Practice 01 — Python and NumPy

## Objective

Get familiar with basic Python and fundamental operations with NumPy to manipulate arrays and vectors.

## Difficulty level

⭐ Basic (L1)

## Estimated duration

30-40 minutes

______________________________________________________________________

## Part 1: Solved (step by step guide)

### Exercise 1.1: Create arrays and perform basic operations

**Instruction:** Create two NumPy arrays with 5 numbers each and perform addition, subtraction, and multiplication element by element.

**Solution:**

```python
import numpy as np

# Create arrays
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# Operaciones elemento a elemento
suma = a + b
resta = b - a
multiplication = a * b

print("Array a:", a)
print("Array b:", b)
print("Suma:", suma)
print("Resta (b-a):", resta)
print("Multiplication:", multiplication)
```

**Output expected:**

```
Array a: [1 2 3 4 5]
Array b: [10 20 30 40 50]
Suma: [11 22 33 44 55]
Resta (b-a): [ 9 18 27 36 45]
Multiplication: [ 10  40  90 160 250]
```

**Explanation:**

- Arithmetic operations in NumPy are applied element by element (broadcasting).
- This is much more efficient than using loops in pure Python.

______________________________________________________________________

### Exercise 1.2: Calculate basic statistics

**Instruction:** Given an array of 10 numbers, calculate the mean, maximum, minimum and total sum.

**Solution:**

```python
import numpy as np

data = np.array([23, 45, 12, 67, 34, 89, 23, 56, 78, 90])

media = np.mean(data)
maximo = np.max(data)
minimo = np.min(data)
suma_total = np.sum(data)

print(f"Data: {data}")
print(f"Media: {media:.2f}")
print(f"Maximum: {maximo}")
print(f"Minimum: {minimo}")
print(f"Suma total: {suma_total}")
```

**Output expected:**

```
Data: [23 45 12 67 34 89 23 56 78 90]
Media: 51.70
Maximum: 90
Minimum: 12
Suma total: 517
```

**Explanation:**

- `np.mean()`, `np.max()`, `np.min()`, `np.sum()` are very fast vectorized functions.
- They are the basis for statistical calculations in large datasets.

______________________________________________________________________

## Part 2: To solve (proposal)

### Exercise 2.1: Filtrado conditional

**Instruction:**
Create an array with 15 random numbers between 1 and 100. Filter and display only numbers greater than 50.

**Tracks:**

- Use `np.random.randint(1, 101, size=15)` to generate random numbers.
- Use boolean indexing: `array[array > 50]`.

**Success Criteria:**

- The original array must have 15 elements.
- The filtered array must only contain values ​​> 50.
- Must be reproducible (set a seed with `np.random.seed(42)`).

______________________________________________________________________

### Exercise 2.2: Reshape and transposition

**Instruction:**
Create an array of 12 consecutive numbers (1 to 12). Reshape to a 3x4 matrix. Then transpose the matrix and display both.

**Tracks:**

- Use `np.arrange(1, 13)` to create consecutive numbers.
- Uses `.reshape(3, 4)` to convert the matrix.
- Use `.T` or `np.transpose()` to transpose.

**Success Criteria:**

- Matrix original: 3 rows x 4 columns.
- Matrix transposed: 4 rows x 3 columns.
- Values ​​must match correctly in transposed positions.

______________________________________________________________________

### Exercise 2.3: Production point entre vectors

**Instruction:**
Create two vectors of size 5 with values ​​of your choice. Calculate:

1. Production point.
1. Norm (magnitude) of each vector.
1. Angle between both vectors (use the cosine formula).

**Tracks:**

- Production point: `np.dot(a, b)`.
- Norma: `np.linalg.norm(a)`.
- Cosine of the angle: `cos(θ) = (a·b) / (||a|| * ||b||)`.
- Angle in degrees: `np.arccos(cosine) * 180 / np.pi`.

**Success Criteria:**

- Production point calculated correctly.
- Norms positive.
- Angle between 0° and 180°.

______________________________________________________________________

## Entregable

- A notebook or script `.py` with the solutions from Part 2.
- Comments explaining each paso.
- Visible outputs (use `print()` to show results).

## Help resources

- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy Array Creation](https://numpy.org/doc/stable/user/basics.creation.html)
- [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
