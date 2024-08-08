import sympy as sp
import numpy as np

# Definir las variables
x, y = sp.symbols('x y')

# Definir el sistema de ecuaciones para F(x, y)
F1 = 3*x**2 - y**2
F2 = 3*x*y**2 - x**3 - 1

# Crear la matriz Jacobiana
F = sp.Matrix([F1, F2])
vars = sp.Matrix([x, y])
J = F.jacobian(vars)

# Convertir a funciones numéricas
F_func = sp.lambdify((x, y), F, 'numpy')
J_func = sp.lambdify((x, y), J, 'numpy')

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
x0 = [0.1, 0.1]

# Ejecutar el método de Newton
approximations, solution = newton_method_sympy(F_func, J_func, x0)

# Mostrar las soluciones con formato específico
print("\nCeros de las funciones:")
print(f"x = {solution[0]:.7e}")
print(f"y = {solution[1]:.7e}")

# Verificar la solución
F1_value = F1.subs({x: solution[0], y: solution[1]})
F2_value = F2.subs({x: solution[0], y: solution[1]})

print("\nVerificación de las soluciones:")
# Imprimir los resultados
print(f"F1 = {F1_value.evalf()}")
print(f"F2 = {F2_value.evalf()}")