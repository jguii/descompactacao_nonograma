from z3 import *
import matplotlib.pyplot as plt
import numpy as np

# Restrições do problema
rows = [[3], [4], [1, 3], [1], [1]]
cols = [[3], [2], [3], [2], [3]]

num_rows = len(rows)
num_cols = len(cols)

# Cria matriz de variáveis booleanas
x = [[Bool(f"x_{i}_{j}") for j in range(num_cols)] for i in range(num_rows)]
s = Solver()

# Gera todas as posições possíveis para blocos
def generate_block_positions(length, blocks):
    n = len(blocks)
    total_block_size = sum(blocks) + (n - 1)
    max_start = length - total_block_size
    positions = []

    def backtrack(pos, acc):
        if pos == n:
            positions.append(acc)
            return
        start = acc[-1] + blocks[pos - 1] + 1 if pos > 0 else 0
        for i in range(start, max_start + start + 1):
            if i + blocks[pos] <= length:
                backtrack(pos + 1, acc + [i])

    backtrack(0, [])
    return positions

# Adiciona as restrições por linha
for i, block in enumerate(rows):
    valid_lines = generate_block_positions(num_cols, block)
    row_constraints = []
    for pos in valid_lines:
        cells = []
        filled = set()
        for b, start in zip(block, pos):
            for j in range(start, start + b):
                filled.add(j)
        for j in range(num_cols):
            if j in filled:
                cells.append(x[i][j])
            else:
                cells.append(Not(x[i][j]))
        row_constraints.append(And(cells))
    s.add(Or(row_constraints))

# Adiciona as restrições por coluna
for j, block in enumerate(cols):
    valid_cols = generate_block_positions(num_rows, block)
    col_constraints = []
    for pos in valid_cols:
        cells = []
        filled = set()
        for b, start in zip(block, pos):
            for i in range(start, start + b):
                filled.add(i)
        for i in range(num_rows):
            if i in filled:
                cells.append(x[i][j])
            else:
                cells.append(Not(x[i][j]))
        col_constraints.append(And(cells))
    s.add(Or(col_constraints))

# Resolve a primeira solução
if s.check() == sat:
    m = s.model()
    grid = [[is_true(m.evaluate(x[i][j])) for j in range(num_cols)] for i in range(num_rows)]
    img = np.array(grid, dtype=int)

    # Cria nova instância do solver para testar unicidade
    s_unique = Solver()
    s_unique.add(s.assertions())

    # Gera a restrição que bloqueia a solução atual
    block_current = []
    for i in range(num_rows):
        for j in range(num_cols):
            val = is_true(m.evaluate(x[i][j]))
            if val:
                block_current.append(Not(x[i][j]))
            else:
                block_current.append(x[i][j])
    s_unique.add(Or(block_current))

    # Verifica se há outra solução
    if s_unique.check() == sat:
        texto = "A imagem não é única"
    else:
        texto = "A imagem é única"

    # Exibe com anotação na imagem
    plt.imshow(img, cmap="Greys", interpolation="nearest")
    plt.axis("off")
    plt.title("Reconstrução a partir da lógica")
    plt.figtext(0.5, 0.01, texto, ha="center", fontsize=12,)
    plt.savefig("imagem_reconstruida.png", bbox_inches='tight')
    plt.show()
else:
    print("Sem solução.")
