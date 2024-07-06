from mpi4py import MPI
import numpy as np
import time

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

class Process:
    def __init__(self, id, num_processes):
        self.id = id
        self.clock = np.zeros(num_processes, dtype=int)
        self.num_processes = num_processes
        self.message_queue = []
        self.processes = {}
    
    def send_multicast_message(self, id_sender, id_receivers, message, send_time, delay):
        receive_time = send_time + delay
        
        for id_receiver in id_receivers:
            packed_message = (id_sender, id_receiver, message, send_time, receive_time)
            
            receiver = self.find_process(id_receiver)
            receiver.receive_message(packed_message)
        
        print(f"Multicast message sent from {id_sender} to {id_receivers} with delay {delay}s")

    def find_process(self, id_receiver):
        return self.processes.get(id_receiver, None)
    
    def receive_message(self, packed_message):
        id_sender, id_receiver, message, send_time, receive_time = packed_message
        m = message
        
        # Condição 1: m[i] = VCj[i] + 1
        condition_1 = m[id_sender] == self.clock[id_sender] + 1

        # Condição 2: m[k] <= VCj[k], para todo k diferente de i
        condition_2 = all(m[k] <= self.clock[k] for k in range(self.num_processes) if k != id_sender)
        
        if condition_1 and condition_2:
            # Atualiza o vetor de causalidade do receptor
            for processo in range(self.num_processes):
                self.clock[processo] = max(self.clock[processo], m[processo])
            
            # Processa a mensagem
            print(f"Process {id_receiver} received message: {message} from Process {id_sender}")
        else:
            # Adiciona a mensagem à fila de espera se as condições não forem satisfeitas
            self.message_queue.append(packed_message)
            print(f"Message from Process {id_sender} to Process {id_receiver} delayed")
        
        # Verifica e processa mensagens na fila de espera que agora podem ser entregues
        self.check_and_process_delayed_messages()
    
    def check_and_process_delayed_messages(self):
        for packed_message in self.message_queue:
            id_sender, id_receiver, message, send_time, receive_time = packed_message
            m = message
            
            # Condição 1: m[i] = VCj[i] + 1
            condition_1 = m[id_sender] == self.clock[id_sender] + 1

            # Condição 2: m[k] <= VCj[k], para todo k diferente de i
            condition_2 = all(m[k] <= self.clock[k] for k in range(self.num_processes) if k != id_sender)
            
            if condition_1 and condition_2:
                # Atualiza o vetor de causalidade do receptor
                for processo in range(self.num_processes):
                    self.clock[processo] = max(self.clock[processo], m[processo])
                
                # Processa a mensagem
                print(f"Delayed message processed: Process {id_receiver} received message: {message} from Process {id_sender}")
                self.message_queue.remove(packed_message)

# Inicialização do MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

num_processes = size
processes = {i: Process(i, num_processes) for i in range(num_processes)}

# Adicionar a lista de processos à classe Process
for process in processes.values():
    process.processes = processes

if __name__ == '__main__':
    if rank == 0:
        time.sleep(1)  # Espera um pouco antes de enviar para garantir que todos os processos estejam prontos
        processes[0].clock[0] += 1
        message = processes[0].clock.copy()
        processes[0].send_multicast_message(0, [1, 2, 3], message, time.time(), 2)
    elif rank == 1:
        time.sleep(3)
        processes[1].clock[1] += 1
        message = processes[1].clock.copy()
        processes[1].send_multicast_message(1, [0, 2, 3], message, time.time(), 1)
    elif rank == 2:
        time.sleep(5)
        processes[2].clock[2] += 1
        message = processes[2].clock.copy()
        processes[2].send_multicast_message(2, [0, 1, 3], message, time.time(), 3)
    elif rank == 3:
        time.sleep(7)
        processes[3].clock[3] += 1
        message = processes[3].clock.copy()
        processes[3].send_multicast_message(3, [0, 1, 2], message, time.time(), 2)
