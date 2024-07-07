from mpi4py import MPI
import numpy as np

""" 
Autor: João Victor Leite da Silva Almeida
Numero de matricula: 202021761
Disciplina: Sistemas Distribuidos
Descricao: Implementacao do algoritmo de vetores de causalidade de Lamport

Como executar:
    - Para baicar as bibliotecas necessarioas:
        pip install -r./requirements.txt
    
    - Para executar o programa:
        mpirun --oversubscribe -np 4 python main.py
"""

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Verificar se o número de processos é 4
if size != 4:
    if rank == 0:
        print("Este programa requer exatamente 4 processos.")
    MPI.Finalize()
    exit()

# Inicializar os vetores de relógios lógicos
vector_clock = np.zeros(size, dtype=int)

def multicast(msg, sender_rank, vc):
    """
    Envia uma mensagem multicast para todos os processos, exceto o remetente.

    Args:
        msg (object): A mensagem a ser enviada.
        sender_rank (int): O índice do processo remetente.
        vc (list): O vetor de relógio vetorial atualizado.

    Returns:
        None
    """
    for i in range(size):
        if i != sender_rank:
            comm.send((msg, vc), dest=i, tag=sender_rank)

def receive_message():
    """
    Função responsável por receber uma mensagem de outros processos e verificar se as condições de recebimento são atendidas.

    Returns:
        msg (objeto): A mensagem recebida.
        sender_rank (int): O rank do processo remetente.
        success (bool): Indica se as condições de recebimento foram atendidas.
    """
    status = MPI.Status()
    msg, sender_vc = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    sender_rank = status.Get_source()
    
    # Condições de recebimento
    # Condição 1: m[i] = VCj[i] + 1
    condition1 = sender_vc[sender_rank] == vector_clock[sender_rank] + 1
    # Condição 2: m[k] <= VCj[k], para todo k diferente de i
    condition2 = all(sender_vc[k] <= vector_clock[k] for k in range(size) if k != sender_rank)
    
    if condition1 and condition2:
        vector_clock[sender_rank] += 1
        return msg, sender_rank, True
    else:
        return msg, sender_rank, False

# Cada processo envia uma mensagem para todos os outros processos
print(f"[SEND]      Processo {rank} enviando mensagem...")
vector_clock[rank] += 1
multicast(f"Mensagem de P{rank}", rank, vector_clock.copy())
print(f"[UPDATE 1]  Processo {rank} vetor atualizado: {vector_clock}")

# Cada processo recebe mensagens de todos os outros processos
received_messages = 0
while received_messages < size - 1:
    msg, sender_rank, received = receive_message()
    if received:
        print(f"[RECEIVE]   Processo {rank} recebeu '{msg}' de P{sender_rank}")
        print(f"[UPDATE 2]  Processo {rank} vetor atualizado: {vector_clock}")
        received_messages += 1
    else:
        print(f"[DELAY]     Processo {rank} não recebeu '{msg}' de P{sender_rank}")


# Sincronizar todos os processos antes de mostrar o vetor final
comm.Barrier()

print(f"[FINAL]     Processo {rank} vetor final: {vector_clock}")
