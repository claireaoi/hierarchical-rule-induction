import torch
import pdb
from torch.autograd import Variable


aviliableTaskName = ['Grandparent', 'Son', 'Connectedness']
aviliableTaskName2Families = [
    # 'TwoChildren',
]


def OriginalTrainingData(task_name):
    if task_name not in aviliableTaskName:
        try:
            raise NameError("Unaviliable task name!!! Aviliable task names are:", aviliableTaskName)
        except NameError:
            raise

    num_constants = 0
    valuation_init = None
    target = None
    if task_name == 'Grandparent':
        num_constants = 9
        ##Background Knowledge
        father_extension = torch.zeros(num_constants, num_constants)
        father_extension[0,1] = 1
        father_extension[0,2] = 1
        father_extension[1,3] = 1
        father_extension[1,4] = 1

        mother_extension = torch.zeros(num_constants, num_constants)
        mother_extension[8,0] = 1
        mother_extension[2,5] = 1
        mother_extension[2,6] = 1
        mother_extension[5,7] = 1

        #Intensional Predicates
        aux_extension = torch.zeros(num_constants, num_constants)
        target_extension = torch.zeros(num_constants, num_constants)

        valuation_init = [Variable(father_extension), Variable(mother_extension),
                        Variable(aux_extension), Variable(target_extension)]
        ##Target
        target = Variable(torch.zeros(num_constants, num_constants))
        target[8,1] =1
        target[8,2] =1
        target[0,3] =1
        target[0,4] =1
        target[0,5] =1
        target[0,6] =1
        target[2,7] =1
    elif task_name == 'Son':
        num_constants = 9
        ##Background Knowledge
        father_extension = torch.zeros(num_constants, num_constants)
        brother_extension = torch.zeros(num_constants, num_constants)
        sister_extension = torch.zeros(num_constants, num_constants)

        father_extension[0,1] = 1 
        father_extension[0,2] = 1 
        father_extension[3,4] = 1 
        father_extension[3,5] =	1
        father_extension[6,7] = 1 
        father_extension[6,8] =	1
        brother_extension[1,2] = 1 
        brother_extension[2,1] = 1
        brother_extension[4,5] = 1 
        sister_extension[5,4] =	1
        sister_extension[7,8] = 1 
        sister_extension[8,7] =	1

        #Intensional Predicates
        aux_extension = torch.zeros(1, num_constants).view(-1, 1)
        son_extension = torch.zeros(num_constants, num_constants)

        ##Target
        target = Variable(torch.zeros(num_constants, num_constants))
        target[1,0] = 1
        target[2,0] = 1
        target[4,3] = 1

        valuation_init = [Variable(father_extension), Variable(brother_extension),
                          Variable(sister_extension), Variable(aux_extension),
                          Variable(son_extension)]
    elif task_name == 'Connectedness':
        num_constants = 4
        ##Background Knowledge
        edge_extension = torch.zeros(num_constants, num_constants)
        edge_extension[0,1] = 1
        edge_extension[1,2] = 1
        edge_extension[2,3] = 1
        edge_extension[1,0] = 1
        #Intensional Predicates
        connected_extension = torch.zeros(num_constants, num_constants)

        ##Target
        target = Variable(torch.zeros(num_constants, num_constants))
        target[0,0] = 1
        target[0,1] = 1
        target[0,2] = 1
        target[0,3] = 1
        target[1,0] = 1
        target[1,1] = 1
        target[1,2] = 1
        target[1,3] = 1
        target[2,3] = 1

        valuation_init = [Variable(edge_extension), Variable(connected_extension)]
        
    return num_constants, valuation_init, target


def OriginalTrainingData2Families(task_name):
    if task_name not in aviliableTaskName2Families:
        try:
            raise NameError("Unaviliable task name!!! Aviliable task names are:", aviliableTaskName2Families)
        except NameError:
            raise

    num_constants = 0
    valuation_init = None
    target = None
    num_constants_2 = 0
    valuation_init_2 = None
    target_2 = None

    if task_name == 'TwoChildren':
        ##Background Knowledge
        num_constants = 5

        neq_extension = torch.ones(num_constants, num_constants)
        for i in range(num_constants):
            neq_extension[i, i] = 0

        edge_extension = torch.zeros(num_constants, num_constants)
        edge_extension[0,1] = 1
        edge_extension[0,2] = 1
        edge_extension[1,3] = 1
        edge_extension[2,3] = 1
        edge_extension[2,4] = 1
        edge_extension[3,4] = 1

        num_constants_2 = 5
        edge_extension_2 = torch.zeros(num_constants_2, num_constants_2)
        edge_extension_2[0,1] = 1
        edge_extension_2[1,2] = 1
        edge_extension_2[1,3] = 1
        edge_extension_2[2,2] = 1
        edge_extension_2[2,4] = 1
        edge_extension_2[3,4] = 1

        #Intensional Predicates
        target_extension = torch.zeros(num_constants, num_constants)
        aux_extension = torch.zeros(num_constants,num_constants)

        valuation_init = [Variable(neq_extension), Variable(edge_extension), Variable(aux_extension), Variable(target_extension)]

        ##Target
        target = Variable(torch.zeros(num_constants,num_constants))
        target[0,0] = 1
        target[2,2] = 1

        #Intensional Predicates
        target_extension_2 = torch.zeros(num_constants_2, num_constants_2)
        aux_extension_2 = torch.zeros(num_constants_2, num_constants_2)

        valuation_init_2 = [Variable(neq_extension), Variable(edge_extension_2), Variable(aux_extension_2), Variable(target_extension_2)]

        ##Target
        target_2 = Variable(torch.zeros(num_constants_2, num_constants_2))
        target_2[1,1] = 1
        target_2[2,2] = 1

        # steps = 2

    return num_constants, valuation_init, target, num_constants_2, valuation_init_2, target_2


def OriginalEvaluationData(task_name):
    if task_name not in aviliableTaskName:
        try:
            raise NameError("Unaviliable task name!!! Aviliable task names are:", aviliableTaskName)
        except NameError:
            raise

    num_constants = 0
    valuation_init = None
    target = None
    if task_name == 'Grandparent':
        try:
            raise NotImplementedError("Haven't finished this part!")
        except NameError:
            raise
    elif task_name == 'Son':
        try:
            raise NotImplementedError("Haven't finished this part!")
        except NameError:
            raise
    elif task_name == 'Connectedness':
        num_constants = 4
        ##Background Knowledge
        edge_extension = torch.zeros(num_constants, num_constants)
        edge_extension[0,1] = 1
        edge_extension[1,2] = 1
        edge_extension[2,3] = 1
        edge_extension[2,0] = 1

        #Intensional Predicates
        connected_extension = torch.zeros(num_constants, num_constants)

        ##Target
        target = Variable(torch.zeros(num_constants, num_constants))
        target[0,0] = 1
        target[0,1] = 1
        target[0,2] = 1
        target[0,3] = 1
        target[1,0] = 1
        target[1,1] = 1
        target[1,2] = 1
        target[1,3] = 1
        target[2,0] = 1
        target[2,1] = 1
        target[2,2] = 1
        target[2,3] = 1

        steps = 5
        valuation_init = [Variable(edge_extension), Variable(connected_extension)]
        
    return num_constants, steps, valuation_init, target
