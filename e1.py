import numpy as np

def F(x):
    """
    Definir el sistema de ecuaciones no lineales.
    """
    return np.array([
        3*x[0] - np.cos(x[1]*x[2]) - 1/2,
        x[0]**2 - 81*(x[1] + 0.1)**2 + np.sin(x[2]) + 1.06,
        np.exp(-x[0]*x[1]) + 20*x[2] + (10*np.pi - 3)/3
    ])

def jacobian(x):
    """
    Definir la matriz Jacobiana del sistema de ecuaciones.
    """
    return np.array([
        [3, x[2]*np.sin(x[1]*x[2]), x[1]*np.sin(x[1]*x[2])],
        [2*x[0], -162*(x[1] + 0.1), np.cos(x[2])],
        [-x[1]*np.exp(-x[0]*x[1]), -x[0]*np.exp(-x[0]*x[1]), 20]
    ])

def newton_method(F, jacobian, x0, tol=1e-7, max_iter=100):
    """
    Método de Newton-Raphson para sistemas no lineales.
    """
    x = x0
    approximations = [x.copy()]
    for i in range(max_iter):
        J = jacobian(x)
        F_x = F(x)
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
x0 = np.array([0.1, 0.1, 0.1])

# Ejecutar el método de Newton
approximations, solution = newton_method(F, jacobian, x0)

# Mostrar las soluciones con formato específico
print("\nResultados finales con precisión de 7 cifras:")
print(f"x1 = {solution[0]:.7e}")
print(f"x2 = {solution[1]:.7e}")
print(f"x3 = {solution[2]:.7e}")