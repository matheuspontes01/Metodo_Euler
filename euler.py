import numpy as np
import matplotlib.pyplot as plt

# Função f(x, y) do PVI -> y' = f(x, y)
def f(x, y):
    # Exemplo: y' = x + y
    return x - y + 2

# Condições iniciais (y(0) = 2)
x0 = 0.0 # ponto inicial
y0 = 2.0 # valor inicial

# malha [0, 1]
a = 0.0 # inicio do intervalo
b = 1.0 # fim do intervalo

h = 0.1       # número de subdivisões

m = int((b - a) / h)  # número de subintervalos

# Inicialização dos vetores x e y
x = np.zeros(m + 1)
y = np.zeros(m + 1)

# Valores iniciais
x[0], y[0] = x0, y0

# Método de Euler
for i in range(m):
    y[i + 1] = y[i] + h * f(x[i], y[i])
    x[i + 1] = x[i] + h

# solução exata: y(x) = x + 1 + e^{-x}
y_exact = 2 * np.exp(x) - x - 1
errors = np.abs(y_exact - y)

# Exibição da tabela
print(" i |   x_i  |  y_i (Euler)  |  y(x_i) exato  |  erro")
print(70*"-")
for i in range(m+1):
    print(f"{i:2d} | {x[i]:4.1f}  | {y[i]:12.6f} | {y_exact[i]:12.6f} | {errors[i]:8.6f}")

# Gráfico comparando as soluções
plt.plot(x, y_exact, 'b-', label='Solução exata y(x)')
plt.plot(x, y, 'ro--', label='Aproximação (Euler)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Método de Euler')
plt.legend()
plt.grid(True)
plt.savefig("grafico_euler.png", dpi=300, bbox_inches='tight')
plt.show()
