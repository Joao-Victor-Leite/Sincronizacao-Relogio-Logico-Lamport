from mpi4py import MPI
import numpy as np
import time

class Process:
    def __init__(self, id, num_processes):
        self.id = id
        self.clock = np.zeros(num_processes, dtype=int)
        self.num_processes = num_processes
        self.message_queue = []
    
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
                self.clock[k] = max(self.clock[processo], m[processo])

            # Processa a mensagem
            print(f"Process {id_receiver} received message: {message} from Process {id_sender}")
        else:
             # Adiciona a mensagem à fila de espera se as condições não forem satisfeitas
             self.message_queue.append(packed_message)
             print(f"Message from Process {id_sender} to Process {id_receiver} delayed")

        # Verifica e processa mensagens na fila de espera que agora podem ser entregues
        self.check_and_process_delayed_messages()

    def check_and_process_delayed_messages():
        ...


# Inicialização do MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

num_processes = size
processes = [Process(i, num_processes) for i in range(num_processes)]

if __name__ == '__main__':
    if rank == 0:
        ...
    elif rank == 1:
        ...
    elif rank == 2:
        ...
    elif rank == 3:
        ...