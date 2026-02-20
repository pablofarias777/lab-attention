# Implementação do Scaled Dot-Product Attention

Este projeto implementa a lógica central do mecanismo de Self-Attention conforme descrito no paper "Attention Is All You Need".

## Fórmula utilizada

Attention(Q, K, V) = softmax((QK^T) / √dk) V

Onde:
- Q = Query
- K = Key
- V = Value
- dk = dimensão das chaves (número de colunas de Q)

A normalização pela raiz quadrada de dk evita que os valores do produto escalar fiquem muito grandes, o que poderia prejudicar a estabilidade do Softmax.

## Como executar

1. Instalar dependências:
pip3 install numpy

2. Rodar o teste:
python3 test_attention.py

## Exemplo de entrada

Q = [[1, 0],
     [0, 1]]

K = [[1, 0],
     [0, 1]]

V = [[1, 2],
     [3, 4]]

## Exemplo de saída

[[1.6604769  2.6604769]
 [2.3395231  3.3395231]]