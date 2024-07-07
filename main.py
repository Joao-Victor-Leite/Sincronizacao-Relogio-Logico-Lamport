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
        self.id = id                                    #O identificador único do processo.
        self.clock = np.zeros(num_processes, dtype=int) #O vetor de relógio lógico do processo.
        self.num_processes = num_processes              #O número total de processos no sistema.
        self.message_queue = []                         #A fila de mensagens recebidas pelo processo.
        self.processes = {}                             #Um dicionário que mapeia os identificadores dos processos aos seus respectivos objetos.
    
    def send_singlecast_message(self, id_sender, id_receiver, message, send_time, delay):
            """
            Envia uma mensagem unicast de um processo remetente para um processo destinatário.

            Parâmetros:
            - id_sender (int): ID do processo remetente.
            - id_receiver (int): ID do processo destinatário.
            - message (list): Lista contendo a mensagem a ser enviada.
            - send_time (float): Tempo de envio da mensagem.
            - delay (float): Tempo de atraso para simular o atraso na entrega da mensagem.

            Retorna:
            Nenhum retorno.
            """
            receive_time = send_time + delay
            packed_message = (id_sender, id_receiver, message.copy(), send_time, receive_time)
            print(f"[SEND] Time: {send_time:.2f} - Process {id_sender} sending message to Process {id_receiver} with delay {delay}s. Message: {message}\n")

            receiver = self.find_process(id_receiver)
            if receiver:
                time.sleep(delay)
                receiver.receive_message(packed_message)


    def find_process(self, id_receiver):
        """
        Retorna o processo correspondente ao ID do receptor.

        Parâmetros:
        - id_receiver (int): O ID do receptor.

        Retorna:
        - O processo correspondente ao ID do receptor, ou None se não for encontrado.
        """
        return self.processes.get(id_receiver, None)
    
    def check_delay_conditions(self, packed_message):
        """
        Verifica as condições de atraso de uma mensagem recebida.

        Parâmetros:
        - packed_message: uma tupla contendo as informações da mensagem recebida, na seguinte ordem:
            - id_sender: o identificador do processo que enviou a mensagem
            - id_receiver: o identificador do processo que recebeu a mensagem
            - message: a mensagem em si
            - send_time: o tempo de envio da mensagem
            - receive_time: o tempo de recebimento da mensagem

        Retorna:
        - True se as condições de atraso forem satisfeitas
        - False caso contrário
        """
        id_sender, id_receiver, message, send_time, receive_time = packed_message
        m = message

        # Condição 1: m[i] = VCj[i] + 1
        condition_1 = m[id_sender] == self.clock[id_sender] + 1

        # Condição 2: m[k] <= VCj[k], para todo k diferente de i
        condition_2 = all(m[k] <= self.clock[k] for k in range(self.num_processes) if k != id_sender)

        if not condition_1 or not condition_2:
            print(f"[DELAY] Time: {receive_time:.2f} - Message from Process {id_sender} to Process {id_receiver} delayed\n")
            return False
        
        return True
    
    def check_and_process_delayed_messages(self):
            """
            Verifica e processa as mensagens atrasadas na fila de mensagens.

            Itera sobre cada mensagem na fila de mensagens e verifica se as condições de atraso são atendidas.
            Se as condições forem atendidas, atualiza o vetor de causalidade do receptor, processa a mensagem
            e remove a mensagem da fila.

            Args:
                self (objeto): A instância da classe.

            Returns:
                None
            """
            for packed_message in list(self.message_queue):
                id_sender, id_receiver, message, send_time, receive_time = packed_message
                m = message
                
                if self.check_delay_conditions(packed_message):
                     # Atualiza o vetor de causalidade do receptor
                    for processo in range(self.num_processes):
                        self.clock[processo] = max(self.clock[processo], m[processo])
                    self.clock[id_receiver] += 1

                    # Processa a mensagem
                    print("Message from Process {} to Process {} delayed".format(id_sender, id_receiver))
                    print(f"[RECEIVE] Time: {receive_time:.2f} - Process {self.id} processed delayed message: {message} from Process {id_sender}\n")
                    print(f"[UPDATE] Time: {receive_time:.2f} - Updated vector clock of Process {self.id} after processing delayed message: {self.clock}\n")
                    self.message_queue.remove(packed_message)


    def receive_message(self, packed_message):
        """
        Recebe uma mensagem e atualiza o relógio vetorial do processo.

        Parâmetros:
        packed_message (tuple): Uma tupla contendo as informações da mensagem recebida.
            A tupla deve conter os seguintes elementos:
                - id_sender (int): O ID do processo que enviou a mensagem.
                - id_receiver (int): O ID do processo que recebeu a mensagem.
                - message (list): Uma lista contendo os valores do relógio vetorial da mensagem.
                - send_time (float): O tempo de envio da mensagem.
                - receive_time (float): O tempo de recebimento da mensagem.

        Retorna:
        None

        """
        id_sender, id_receiver, message, send_time, receive_time = packed_message

        if self.check_delay_conditions(packed_message):
            for processo in range(len(self.clock)):
                self.clock[processo] = max(self.clock[processo], message[processo])
            self.clock[id_receiver] += 1
            print(f"[RECEIVE] Time: {receive_time:.2f} - Process {self.id} received message: {message} from Process {id_sender}\n")
            print(f"[UPDATE] Time: {receive_time:.2f} - Updated vector clock of Process {self.id}: {self.clock}\n")
        else:
            self.message_queue.append(packed_message)
        
        self.check_and_process_delayed_messages()


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
        processes[0].send_singlecast_message(0, 1, message, time.time(), 2)
        processes[0].send_singlecast_message(0, 2, message, time.time(), 3)
        processes[0].send_singlecast_message(0, 3, message, time.time(), 4)
    elif rank == 1:
        time.sleep(3)
        processes[1].clock[1] += 1
        message = processes[1].clock.copy()
        processes[1].send_singlecast_message(1, 0, message, time.time(), 1)
        processes[1].send_singlecast_message(1, 2, message, time.time(), 2)
        processes[1].send_singlecast_message(1, 3, message, time.time(), 3)
    elif rank == 2:
        time.sleep(5)
        processes[2].clock[2] += 1
        message = processes[2].clock.copy()
        processes[2].send_singlecast_message(2, 0, message, time.time(), 2)
        processes[2].send_singlecast_message(2, 1, message, time.time(), 3)
        processes[2].send_singlecast_message(2, 3, message, time.time(), 4)
    elif rank == 3:
        time.sleep(7)
        processes[3].clock[3] += 1
        message = processes[3].clock.copy()
        processes[3].send_singlecast_message(3, 0, message, time.time(), 1)
        processes[3].send_singlecast_message(3, 1, message, time.time(), 2)
        processes[3].send_singlecast_message(3, 2, message, time.time(), 3)

    comm.Barrier()

    # Coletar os vetores finais de todos os processos no processo 0
    final_clocks = comm.gather(processes[rank].clock, root=0)

    # Imprimir o vetor de causalidade final de cada processo no processo 0
    if rank == 0:
        print("\n[FINAL] Vetores de causalidade finais de todos os processos:")
        for i, clock in enumerate(final_clocks):
            print(f"Process {i} final vector clock: {clock}")

    comm.Barrier()

    # Finalização do MPI
    MPI.Finalize()