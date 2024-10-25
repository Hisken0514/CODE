import numpy as np

# 定義 Level 3 BLAS 操作
def level3_blas(operation, A, B, C, a=1, b=0):
   
    if operation == 'GEMM':
        C = a * np.dot(A, B) + b * C
    elif operation == 'SYMM':
        C = a * (np.dot(A, B) + np.dot(B, A)) + b * C
    elif operation == 'SYRK':
        C = a * np.dot(A, A.T) + b * C  
    elif operation == 'SYR2K':
        C = a * (np.dot(A, B.T) + np.dot(B, A.T)) + b * C  
    elif operation == 'TRMM':
        C = a * np.dot(A, B)  
    elif operation == 'TRSM':
        C = a * np.dot(np.linalg.inv(A), B)  
    else:
        raise ValueError("Unknown operation")
    
    return C

# 示例數據
A = np.random.randint(0, 10, size=(10, 10))  # 生成3x3的整數矩陣
B = np.random.randint(0, 10, size=(10, 10))  # 隨機生成3x3矩陣B
C = np.random.randint(0, 10, size=(10, 10))  # 隨機生成3x3矩陣C

# 執行不同的 Level 3 BLAS 操作
gemm_result = level3_blas('GEMM', A, B, C.copy(), a=1, b=1)
symm_result = level3_blas('SYMM', A, B, C.copy(), a=1, b=1)
syrk_result = level3_blas('SYRK', A, A, C.copy(), a=1, b=1)  # 用於SYRK，B應與A相同
syr2k_result = level3_blas('SYR2K', A, B, C.copy(), a=1, b=1)
trmm_result = level3_blas('TRMM', A, B, C.copy(), a=1, b=0)  # B作為輸出
trsm_result = level3_blas('TRSM', A, B, C.copy(), a=1, b=0)  # 解方程Ax = B

print(f'A = \n{A}')  
print(f'B = \n{B}')  
print(f'C = \n{C}')  
