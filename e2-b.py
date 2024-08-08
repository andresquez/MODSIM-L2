import sympy as sp
import numpy as np

# Definir las variables
x, y, z = sp.symbols('x y z')

# Definir el sistema de ecuaciones para G(x, y, z)
G1 = 12*x - 3*y**2 - 4*z - 7.17
G2 = x + 10*y - z - 11.54
G3 = y**3 - 7*z**3 - 7.631

# Crear la matriz Jacobiana
G = sp.Matrix([G1, G2, G3])
vars = sp.Matrix([x, y, z])
J = G.jacobian(vars)

# Convertir a funciones numéricas
G_func = sp.lambdify((x, y, z), G, 'numpy')
J_func = sp.lambdify((x, y, z), J, 'numpy')

def newton_method_sympy(G_func, J_func, x0, tol=1e-7, max_iter=100):
    """
    Método de Newton-Raphson para sistemas no lineales usando sympy.
    """
    x = np.array(x0, dtype=float)
    approximations = [x.copy()]
    for i in range(max_iter):
        J = np.array(J_func(*x), dtype=float)
        G_x = np.array(G_func(*x), dtype=float).flatten()
        try:
            delta_x = np.linalg.solve(J, -G_x)
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
approximations, solution = newton_method_sympy(G_func, J_func, x0)

# Mostrar las soluciones con formato específico
print("\nCeros de la funciones:")
print(f"x = {solution[0]:.7e}")
print(f"y = {solution[1]:.7e}")
print(f"z = {solution[2]:.7e}")

# Verificar la solución
G1_value = G1.subs({x: solution[0], y: solution[1], z: solution[2]})
G2_value = G2.subs({x: solution[0], y: solution[1], z: solution[2]})
G3_value = G3.subs({x: solution[0], y: solution[1], z: solution[2]})

print("\nVerificación de las soluciones:")
# Imprimir los resultados
print(f"G1 = {G1_value.evalf()}")
print(f"G2 = {G2_value.evalf()}")
print(f"G3 = {G3_value.evalf()}")