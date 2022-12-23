"""Colocar o nome dos elementos do grupo"""

import random
import math

#valor exemplificativo
alpha = 0.2

def make(nx, nz, ny):
    """Funcao que cria, inicializa e devolve uma rede neuronal, incluindo
    a criacao das diversos listas, bem como a inicializacao das listas de pesos. 
    Note-se que sao incluidas duas unidades extra, uma de entrada e outra escondida, 
    mais os respectivos pesos, para lidar com os tresholds; note-se tambem que, 
    tal como foi discutido na teorica, as saidas destas estas unidades estao sempre a -1.
    por exemplo, a chamada make(3, 5, 2) cria e devolve uma rede 3x5x2"""
    
    
    #a rede neuronal é num dicionario com as seguintes chaves:
    # nx     numero de entradas
    # nz     numero de unidades escondidas
    # ny     numero de saidas
    # x      lista de armazenamento dos valores de entrada
    # z      array de armazenamento dos valores de activacao das unidades escondidas
    # y      array de armazenamento dos valores de activacao das saidas
    # wzx    array de pesos entre a camada de entrada e a camada escondida
    # wyz    array de pesos entre a camada escondida e a camada de saida
    # dz     array de erros das unidades escondidas
    # dy     array de erros das unidades de saida    
    
    nn = {'nx':nx, 'nz':nz, 'ny':ny, 'x':[], 'z':[], 'y':[], 'wzx':[], 'wyz':[], 'dz':[], 'dy':[]}
    
    nn['wzx'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nx'] + 1)] for _ in range(nn['nz'])]
    nn['wyz'] = [[random.uniform(-0.5,0.5) for _ in range(nn['nz'] + 1)] for _ in range(nn['ny'])]
    
    return nn

def sig(input):
    """Funcao de activacao (sigmoide)"""
    return 1.0/(1.0 + math.exp(- input))


def forward(nn, input):
    """Função que recebe uma rede nn e um padrao de entrada in (uma lista) 
    e faz a propagacao da informacao para a frente ate as saidas"""
    
    #copia a informacao do vector de entrada in para a listavector de inputs da rede nn  
    nn['x']=input.copy()
    nn['x'].append(-1)
    
    #calcula a activacao da unidades escondidas
    nn['z']=[sig(sum([x*w for x, w in zip(nn['x'], nn['wzx'][i])])) for i in range(nn['nz'])]
    nn['z'].append(-1)
    
    #calcula a activacao da unidades de saida
    nn['y']=[sig(sum([z*w for z, w in zip(nn['z'], nn['wyz'][i])])) for i in range(nn['ny'])]
 
   
def error(nn, output):
    """Funcao que recebe uma rede nn com as activacoes calculadas
       e a lista output de saidas pretendidas e calcula os erros
       na camada escondida e na camada de saida"""
    
    nn['dy']=[y*(1-y)*(o-y) for y,o in zip(nn['y'], output)]
    
    zerror=[sum([nn['wyz'][i][j]*nn['dy'][i] for i in range(nn['ny'])]) for j in range(nn['nz'])]
    
    nn['dz']=[z*(1-z)*e for z, e in zip(nn['z'], zerror)]
 
 
def update(nn):
    """funcao que recebe uma rede com as activacoes e erros calculados e
    actualiza as listas de pesos"""
    
    nn['wzx'] = [[w+x*nn['dz'][i]*alpha for w, x in zip(nn['wzx'][i], nn['x'])] for i in range(nn['nz'])]
    nn['wyz'] = [[w+z*nn['dy'][i]*alpha for w, z in zip(nn['wyz'][i], nn['z'])] for i in range(nn['ny'])]
    

def iterate(i, nn, input, output):
    """Funcao que realiza uma iteracao de treino para um dado padrao de entrada input
    com saida desejada output"""
    
    forward(nn, input)
    error(nn, output)
    update(nn)
    print('%03i: %s -----> %s : %s' %(i, input, output, nn['y']))
    


def train_and():
    """Funcao que cria uma rede 2x2x1 e treina um AND"""
    
    net = make(2, 2, 1)
    for i in range(2000):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [0])
        iterate(i, net, [1, 0], [0])
        iterate(i, net, [1, 1], [1])
    return net
    
def train_or():
    """Funcao que cria uma rede 2x2x1 e treina um OR"""
    
    net = make(2, 2, 1)
    for i in range(1000):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [1]) 
    return net

def train_xor():
    """Funcao que cria uma rede 2x2x1 e treina um XOR"""
    
    net = make(2, 2, 1)
    for i in range(10000):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [0]) 
    return net
    


def run():
    """Funcao principal do nosso programa, cria os conjuntos de treino e teste, chama
    a funcao que cria e treina a rede e, por fim, a funcao que a treina"""

#conjunto de treino
    build_sets('zoo.txt')[0]

#conjunto de teste
    build_sets('zoo.txt')[1]


    test_zoo(train_zoo(build_sets('zoo.txt')[0]), build_sets('zoo.txt')[1])
    update(train_zoo(build_sets('zoo.txt')[0]))
    test_zoo(train_zoo(build_sets('zoo.txt')[0]), build_sets('zoo.txt')[1])


    


def build_sets(f):
    """Funcao que cria os conjuntos de treino e de de teste a partir dos dados
    armazenados em f (zoo.txt). A funcao le cada linha, tranforma-a numa lista
    de valores e chama a funcao translate para a colocar no formato adequado para
    o padrao de treino. Estes padroes são colocados numa lista 
    Finalmente, devolve duas listas, uma com os primeiros 67 padroes (conjunto de treino)
    e a segunda com os restantes (conjunto de teste)"""
    lista_padroes = []
    linha = ''
    with open(f, 'r') as f:
        for line in f:
            linha = line[0:len(line)-1].strip("[]").split(',')
            for i in range(len(linha)):
                if linha[i].isdigit():
                    linha[i] = int(linha[i])
            lista_padroes+=translate(linha)

    random.shuffle(lista_padroes)

    conjunto_de_treino = lista_padroes[:67]
    conjunto_de_teste =lista_padroes[67:]

    return  conjunto_de_treino, conjunto_de_teste


def translate(lista):
    """Recebe cada lista de valores e transforma-a num padrao de treino.
    Cada padrao tem o formato [nome_do_animal, padrao_de_entrada, tipo_do_animal, padrao_de_saida].
    nome_do_animal e o primeiro valor da lista e tipo_de_animal o ultimo.
    padrao_de_entrada e uma lista de 0 e 1 com os valores dos atributos.
    O numero de pernas deve tambem ser convertido numa lista de 0 e 1, concatenada com os restantes
    atributos. E.g. [0 0 0 0 1 0 0 0 0 0] -> 4 pernas.
    padrao_de_saida e uma lista de 0 e 1 que representa o tipo do animal. Tem 7 posicoes e a unica
    que estiver a 1 corresponde ao tipo do animal. E.g., [0 0 1 0 0 0 0] -> reptile.
    """
    padrao_treino = []
    padrao_de_entrada =lista[1:17]# lista com lista de atributos
    legs = [0,0,0,0,0,0,0,0,0,0]
    tipos=['mammal', 'bird', 'reptile', 'fish', 'amphibian', 'insect', 'invertebrate']
    padrao_de_saida=[0,0,0,0,0,0,0]


      # PERNAS
    for j in range(len(legs)):
        if j == padrao_de_entrada[12]:  # qd index do legs = ao numero do atributo(numero de pernas) mete essa posiçao a 1
                legs[j] = 1
    padrao_de_entrada = padrao_de_entrada[:12] + legs + padrao_de_entrada[13:]


    for i in range(len(tipos)):
        if tipos[i] == lista[17]:
            padrao_de_saida[i] = 1
    padrao_treino.append([lista[0], padrao_de_entrada, lista[17], padrao_de_saida])


    return padrao_treino
    #pass
def train_zoo(training_set):
    """cria a rede e chama a funçao iterate para a treinar. Use 300 iteracoes"""
    nz = math.ceil(math.sqrt(25*7))
    net = make(25, nz, 7)
    for i in range(300):
        for j in range(len(training_set)):
            iterate(i, net, training_set[j][1], training_set[j][3])
    return net

def retranslate(out):
    """recebe o padrao de saida da rede e devolve o tipo de animal corresponte.
    Devolve o tipo de animal corresponde ao indice da saida com maior valor."""
    index= out.index(max(out))
    tipos=['mammal', 'bird', 'reptile', 'fish', 'amphibian', 'insect', 'invertebrate']
    return tipos[index]

def test_zoo(net, test_set):
    """Funcao que avalia a precisao da rede treinada, utilizando o conjunto de teste.
    Para cada padrao do conjunto de teste chama a funcao forward e determina o tipo
    do animal que corresponde ao maior valor da lista de saida. O tipo determinado
    pela rede deve ser comparado com o tipo real, sendo contabilizado o número
    de respostas corretas. A função calcula a percentagem de respostas corretas"""
    count =0
    for padrao in test_set:
        forward(net, padrao[1])
        retranslate(net['y'])
        if retranslate(net['y']) == padrao[2]:
            count+=1
            print("The network thinks ", padrao[0], " is a ", retranslate(net['y']), " and it is correct")

    print("Success rate: ", count/len(test_set)*100, "%")




if __name__ == "__main__":
    #train_and()
    #train_or()
    #train_xor()
    #print(build_sets('zoo.txt'))
    #train_zoo(build_sets('zoo.txt'))
    #test_zoo(train_zoo(build_sets('zoo.txt')[0]), build_sets('zoo.txt')[1])




    run()