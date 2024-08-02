import sympy as sp
import numpy as np

# Definir las variables
x1, x2, x3 = sp.symbols('x1 x2 x3')

# Definir el sistema de ecuaciones
F1 = 3*x1 - sp.cos(x2*x3) - 1/2
F2 = 2*x1 - 81*(x2 + 0.1)**2 + sp.sin(x3) + 1.06
F3 = sp.exp(-x1*x2) + 20*x3 + (10*sp.pi - 3)/3

# Crear la matriz Jacobiana
F = sp.Matrix([F1, F2, F3])
vars = sp.Matrix([x1, x2, x3])
J = F.jacobian(vars)

# Convertir a funciones numéricas
F_func = sp.lambdify((x1, x2, x3), F, 'numpy')
J_func = sp.lambdify((x1, x2, x3), J, 'numpy')

def newton_method_sympy(F_func, J_func, x0, tol=1e-7, max_iter=100):
    """
    Método de Newton-Raphson para sistemas no lineales usando sympy.
    """
    x = np.array(x0, dtype=float)
    approximations = [x.copy()]
    for i in range(max_iter):
        J = np.array(J_func(*x), dtype=float)
        F_x = np.array(F_func(*x), dtype=float).flatten()
        try:
            delta_x = np.linalg.solve(J, -F_x)
        except np.linalg.LinAlgError:
            raise ValueError("Matriz Jacobiana singular. No se puede invertir.")
        
        x = x + delta_x
        approximations.append(x.copy())
        
        # Mostrar la iteración
        print(f"Iteración {i + 1}: x = {x}")
        
        if np.linalg.norm(delta_x, ord=np.inf) < tol:
            break

    return approximations, x

# Punto inicial
x0 = [0.1, 0.1, 0.1]

# Ejecutar el método de Newton
approximations, solution = newton_method_sympy(F_func, J_func, x0)

# Mostrar las soluciones con formato específico
print("\nResultados finales con precisión de 7 cifras:")
print(f"x1 = {solution[0]:.7e}")
print(f"x2 = {solution[1]:.7e}")
print(f"x3 = {solution[2]:.7e}")

# Verificar la solución

# Definición de variables simbólicas
x1, x2, x3 = sp.symbols('x1 x2 x3')

# Soluciones obtenidas
solution = [solution[0], solution[1], solution[2]]

# Evaluar las ecuaciones con las soluciones obtenidas
F1_value = F1.subs({x1: solution[0], x2: solution[1], x3: solution[2]})
F2_value = F2.subs({x1: solution[0], x2: solution[1], x3: solution[2]})
F3_value = F3.subs({x1: solution[0], x2: solution[1], x3: solution[2]})

print("\nVerificación de las soluciones:")
# Imprimir los resultados
print(f"F1 = {F1_value.evalf()}")
print(f"F2 = {F2_value.evalf()}")
print(f"F3 = {F3_value.evalf()}")