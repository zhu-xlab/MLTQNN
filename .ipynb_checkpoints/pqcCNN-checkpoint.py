import tensorflow as tf
import tensorflow_quantum as tfq
import sympy
import cirq
from cirq.contrib.svg import SVGCircuit
from cirq.circuits.qasm_output import QasmUGate
from tensorflow_quantum.python.layers.circuit_executors import expectation, sampled_expectation
from tensorflow_quantum.python.layers.circuit_construction import elementary
from tensorflow_quantum.python import util
import numpy as np

def kernel_prepare(circ, xloc, yloc, target, kernel, readout, symbols0, ctrl):
    loc_states = [[0, 0], [1, 0], [0, 1], [1, 1]]
    for i, loc_state in enumerate(loc_states):
        ctrl_state = loc_state + [1] + ctrl
        circ.append(cirq.rx(symbols0[3 * i + 0]).on(readout).controlled_by(yloc[0], xloc[0], target, kernel[0], control_values=ctrl_state))
        circ.append(cirq.ry(symbols0[3 * i + 1]).on(readout).controlled_by(yloc[0], xloc[0], target, kernel[0], control_values=ctrl_state))
        circ.append(cirq.rz(symbols0[3 * i + 2]).on(readout).controlled_by(yloc[0], xloc[0], target, kernel[0], control_values=ctrl_state))
    return circ


def conv_layer(circ, xloc, yloc, target, kernel, readout, symbols0):
    ctrls = [[0], [1]]
    for i, ctrl in enumerate(ctrls):
        circ = kernel_prepare(circ, xloc, yloc, target, kernel, readout, symbols0[12 * i:12 * i + 12], ctrl)
    return circ

def readout(read_type, qubits):
    output, imgs, kernels, channels = [], [], [], []
    loc1, loc2, readout, kernel, entangle1, entangle2, entangle3 = qubits[2], qubits[5], qubits[11], qubits[9], qubits[6], qubits[7], qubits[8]
   
    if read_type == 'allEntM':
        imgs.append((1 + cirq.X(loc1)) * (1 + cirq.X(loc2)) * (1 - cirq.X(readout)))
        imgs.append((1 - cirq.X(loc1)) * (1 + cirq.X(loc2)) * (1 - cirq.X(readout)))
        imgs.append((1 + cirq.X(loc1)) * (1 - cirq.X(loc2)) * (1 - cirq.X(readout)))
        imgs.append((1 - cirq.X(loc1)) * (1 - cirq.X(loc2)) * (1 - cirq.X(readout)))

        kernels.append((1 + cirq.X(kernel)))
        kernels.append((1 - cirq.X(kernel)))

        channels.append((1 - cirq.X(entangle1)) * (1 - cirq.X(entangle2)) * (1 - cirq.X(entangle3)))
        channels.append((1 + cirq.X(entangle1)) * (1 - cirq.X(entangle2)) * (1 - cirq.X(entangle3)))
        channels.append((1 - cirq.X(entangle1)) * (1 + cirq.X(entangle2)) * (1 - cirq.X(entangle3)))
        channels.append((1 + cirq.X(entangle1)) * (1 + cirq.X(entangle2)) * (1 - cirq.X(entangle3)))
        channels.append((1 - cirq.X(entangle1)) * (1 - cirq.X(entangle2)) * (1 + cirq.X(entangle3)))
        channels.append((1 + cirq.X(entangle1)) * (1 - cirq.X(entangle2)) * (1 + cirq.X(entangle3)))
        channels.append((1 - cirq.X(entangle1)) * (1 + cirq.X(entangle2)) * (1 + cirq.X(entangle3)))
        channels.append((1 + cirq.X(entangle1)) * (1 + cirq.X(entangle2)) * (1 + cirq.X(entangle3)))

    if read_type == 'singleEntM':
        imgs.append((1 + cirq.X(loc1)) * (1 + cirq.X(loc2)) * (1 - cirq.X(readout)))
        imgs.append((1 - cirq.X(loc1)) * (1 + cirq.X(loc2)) * (1 - cirq.X(readout)))
        imgs.append((1 + cirq.X(loc1)) * (1 - cirq.X(loc2)) * (1 - cirq.X(readout)))
        imgs.append((1 - cirq.X(loc1)) * (1 - cirq.X(loc2)) * (1 - cirq.X(readout)))

        kernels.append((1 + cirq.X(kernel)))
        kernels.append((1 - cirq.X(kernel)))

        channels.append((1 - cirq.X(entangle3)))
        channels.append((1 + cirq.X(entangle3)))

    for img in imgs:
        for kernel in kernels:
            for channel in channels:
                output.append(img*kernel*channel)        
    return output


# class allEntEncodingPQC(tf.keras.layers.Layer):
#     def __init__(self, name="PQC"):
#         super(allEntEncodingPQC, self).__init__(name=name)
#         self.flatten = tf.keras.layers.Flatten()
        
#         nElements = 9
#         nQubits = 12

#         qubits = cirq.GridQubit.rect(1, nQubits)
#         observables = readout('allEntM', qubits)

#         inputs = sympy.symbols(f'img:{64*nElements}')
#         params_conv = sympy.symbols(f'conv:{144}')
#         params_loc = sympy.symbols(f'loc:{63}')

#         circ = cirq.Circuit()
#         circ = self.encoding(qubits, circ, inputs)
#         circ = self.qdcnn(qubits, circ, params_conv, params_loc, 1)

#         self.conv = self.add_weight('conv', shape=(1, len(list(params_conv))),
#                                     initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
#                                     trainable=True)
        
#         self.loc_p = self.add_weight('loc_p', shape=(1, len(list(params_loc))),
#                                     initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
#                                     trainable=True)

#         self.symbols = [str(symb) for symb in list(inputs) + list(params_loc) + list(params_conv)]

#         self.circ = tfq.convert_to_tensor([circ])
#         self.operators = tfq.convert_to_tensor([observables])
#         self.executor = expectation.Expectation(backend='noiseless', differentiator=None)
        
#     def encoding(self, qubits, circ, inputs):
#         loc = qubits[:6]
#         target = qubits[6:9]
#         circ.append(cirq.H.on_each(loc))
                
#         for i in range(8):
#             for j in range(8):
#                 row = [int(binary) for binary in format(i, '03b')]
#                 column = [int(binary) for binary in format(j, '03b')]
#                 ctrl_state = row + column
#                 for n in range(len(target)):
#                     circ.append(cirq.rx(inputs[9*(8*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
#                     circ.append(cirq.ry(inputs[9*(8*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
#                     circ.append(cirq.rz(inputs[9*(8*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))

#                 circ.append(cirq.CZ(target[0], target[1]))
#                 circ.append(cirq.CZ(target[1], target[2]))
#                 circ.append(cirq.CZ(target[2], target[0]))

#         return circ

#     def loc_conv(self, qubit1, qubit2, qubit3, circ, symbols):
#         circ.append(cirq.ry(symbols[0]).on(qubit1))
        
#         circ.append(cirq.ry(symbols[1]).on(qubit2).controlled_by(qubit1, control_values=[0]))
#         circ.append(cirq.ry(symbols[2]).on(qubit2).controlled_by(qubit1, control_values=[1]))

#         circ.append(cirq.ry(symbols[3]).on(qubit3).controlled_by(qubit1, qubit2, control_values=[0,0]))
#         circ.append(cirq.ry(symbols[4]).on(qubit3).controlled_by(qubit1, qubit2, control_values=[1,0]))
#         circ.append(cirq.ry(symbols[5]).on(qubit3).controlled_by(qubit1, qubit2, control_values=[0,1]))
#         circ.append(cirq.ry(symbols[6]).on(qubit3).controlled_by(qubit1, qubit2, control_values=[1,1]))  
#         return circ

        
#     def qdcnn(self, qubits, circ, symbols, paras, nQconv):
#         loc = qubits[:6]
#         color = qubits[6:9]
#         kernel = qubits[9:10]
#         readout = qubits[10:12]
#         circ.append(cirq.H.on_each(kernel))
#         for i in range(nQconv):
#             for j in range(len(color)):
#                 circ = self.loc_conv(qubits[3], qubits[0], kernel[0], circ, paras[63*i+21*j+0:63*i+21*j+7])
#                 circ = conv_layer(circ, qubits[3:6], qubits[0:3], color[j], kernel, readout[0], symbols[i*144+48*j:i*144+48*j+24])
#                 circ = self.loc_conv(qubits[4], qubits[1], kernel[0], circ, paras[63*i+21*j+7:63*i+21*j+14])
#                 circ = conv_layer(circ, qubits[4:6], qubits[1:3], readout[0], kernel, readout[1], symbols[i*144+48*j+24:i*144+48*j+48])        
#                 circ = self.loc_conv(qubits[5], qubits[2], kernel[0], circ, paras[63*i+21*j+14:63*i+21*j+21])
        
#         return circ
        
#     def call(self, inputs):
#         inputs = self.flatten(inputs)
#         batch_dim = tf.gather(tf.shape(inputs), 0)
#         tiled_up_circuits = tf.repeat(self.circ, repeats=batch_dim)
#         tiled_up_operators = tf.tile(self.operators, multiples=[batch_dim, 1])
#         tiled_up_conv = tf.tile(self.conv, multiples=[batch_dim, 1])
#         tiled_up_loc_p = tf.tile(self.loc_p, multiples=[batch_dim, 1])

#         joined_vars = tf.concat([inputs, tiled_up_loc_p, tiled_up_conv], axis=1)

#         return self.executor(tiled_up_circuits, symbol_names=self.symbols, symbol_values=joined_vars, operators=tiled_up_operators)


class allEntEncodingPQC_u3loc(tf.keras.layers.Layer):
    def __init__(self, name="PQC"):
        super(allEntEncodingPQC_u3loc, self).__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        
        nElements = 9
        nQubits = 12

        qubits = cirq.GridQubit.rect(1, nQubits)
        observables = readout('allEntM', qubits)

        inputs = sympy.symbols(f'img:{64*nElements}')
        params_conv = sympy.symbols(f'conv:{144}')
        params_loc = sympy.symbols(f'loc:{54}')

        circ = cirq.Circuit()
        circ = self.encoding(qubits, circ, inputs)
        circ = self.qdcnn(qubits, circ, params_conv, params_loc, 1)

        self.conv = self.add_weight('conv', shape=(1, len(list(params_conv))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        
        self.loc_p = self.add_weight('loc_p', shape=(1, len(list(params_loc))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)

        self.symbols = [str(symb) for symb in list(inputs) + list(params_loc) + list(params_conv)]

        self.circ = tfq.convert_to_tensor([circ])
        self.operators = tfq.convert_to_tensor([observables])
        self.executor = expectation.Expectation(backend='noiseless', differentiator=None)
        
    def encoding(self, qubits, circ, inputs):
        loc = qubits[:6]
        target = qubits[6:9]
        circ.append(cirq.H.on_each(loc))
                
        for i in range(8):
            for j in range(8):
                row = [int(binary) for binary in format(i, '03b')]
                column = [int(binary) for binary in format(j, '03b')]
                ctrl_state = row + column
                for n in range(len(target)):
                    circ.append(cirq.rx(inputs[9*(8*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                    circ.append(cirq.ry(inputs[9*(8*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                    circ.append(cirq.rz(inputs[9*(8*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))

                circ.append(cirq.CZ(target[0], target[1]))
                circ.append(cirq.CZ(target[1], target[2]))
                circ.append(cirq.CZ(target[2], target[0]))

        return circ

    def loc_conv(self, qubit1, qubit2, circ, symbols):

        circ.append(cirq.rx(symbols[0]).on(qubit1))
        circ.append(cirq.ry(symbols[1]).on(qubit1))
        circ.append(cirq.rz(symbols[2]).on(qubit1)) 
        
        
        circ.append(cirq.rx(symbols[3]).on(qubit2))
        circ.append(cirq.ry(symbols[4]).on(qubit2))
        circ.append(cirq.rz(symbols[5]).on(qubit2))                
        
        return circ

        
    def qdcnn(self, qubits, circ, symbols, paras, nQconv):
        loc = qubits[:6]
        color = qubits[6:9]
        kernel = qubits[9:10]
        readout = qubits[10:12]
        circ.append(cirq.H.on_each(kernel))
        for i in range(nQconv):
            for j in range(len(color)):
                circ = self.loc_conv(qubits[3], qubits[0], circ, paras[54*i+18*j+0:54*i+18*j+6])
                circ = conv_layer(circ, qubits[3:6], qubits[0:3], color[j], kernel, readout[0], symbols[i*144+48*j:i*144+48*j+24])
                circ = self.loc_conv(qubits[4], qubits[1], circ, paras[54*i+18*j+6:54*i+18*j+12])
                circ = conv_layer(circ, qubits[4:6], qubits[1:3], readout[0], kernel, readout[1], symbols[i*144+48*j+24:i*144+48*j+48])   
                circ = self.loc_conv(qubits[5], qubits[2], circ, paras[54*i+18*j+12:54*i+18*j+18])
        
        return circ
        
    def call(self, inputs):
        inputs = self.flatten(inputs)
        batch_dim = tf.gather(tf.shape(inputs), 0)
        tiled_up_circuits = tf.repeat(self.circ, repeats=batch_dim)
        tiled_up_operators = tf.tile(self.operators, multiples=[batch_dim, 1])
        tiled_up_conv = tf.tile(self.conv, multiples=[batch_dim, 1])
        tiled_up_loc_p = tf.tile(self.loc_p, multiples=[batch_dim, 1])

        joined_vars = tf.concat([inputs, tiled_up_loc_p, tiled_up_conv], axis=1)

        return self.executor(tiled_up_circuits, symbol_names=self.symbols, symbol_values=joined_vars, operators=tiled_up_operators)
    

class allEntEncodingPQC_pureU3loc(tf.keras.layers.Layer):
    def __init__(self, name="PQC"):
        super(allEntEncodingPQC_pureU3loc, self).__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        
        nElements = 9
        nQubits = 12

        qubits = cirq.GridQubit.rect(1, nQubits)
        observables = readout('allEntM', qubits)

        loc_u3 = sympy.symbols(f'locU:{18}')        
        inputs = sympy.symbols(f'img:{64*nElements}')
        params_conv = sympy.symbols(f'conv:{144}')
        params_loc = sympy.symbols(f'loc:{54}')

        circ = cirq.Circuit()
        circ = self.loc(qubits[:6], circ, loc_u3)
        circ = self.encoding(qubits, circ, inputs)
        circ = self.qdcnn(qubits, circ, params_conv, params_loc, 1)
        
        self.loc_u3 = self.add_weight('loc_u3', shape=(1, len(list(loc_u3))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)

        self.conv = self.add_weight('conv', shape=(1, len(list(params_conv))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        
        self.loc_p = self.add_weight('loc_p', shape=(1, len(list(params_loc))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)

        self.symbols = [str(symb) for symb in list(loc_u3) + list(inputs) + list(params_loc) + list(params_conv)]

        self.circ = tfq.convert_to_tensor([circ])
        self.operators = tfq.convert_to_tensor([observables])
        self.executor = expectation.Expectation(backend='noiseless', differentiator=None)
        
    def encoding(self, qubits, circ, inputs):
        loc = qubits[:6]
        target = qubits[6:9]
                
        for i in range(8):
            for j in range(8):
                row = [int(binary) for binary in format(i, '03b')]
                column = [int(binary) for binary in format(j, '03b')]
                ctrl_state = row + column
                for n in range(len(target)):
                    circ.append(cirq.rx(inputs[9*(8*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                    circ.append(cirq.ry(inputs[9*(8*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                    circ.append(cirq.rz(inputs[9*(8*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))

                circ.append(cirq.CZ(target[0], target[1]))
                circ.append(cirq.CZ(target[1], target[2]))
                circ.append(cirq.CZ(target[2], target[0]))

        return circ

    def loc(self, qubits, circ, symbols):
        for i, qubit in enumerate(qubits):
            circ.append(cirq.rx(symbols[3*i+0]).on(qubit))
            circ.append(cirq.ry(symbols[3*i+1]).on(qubit))
            circ.append(cirq.rz(symbols[3*i+2]).on(qubit)) 
                 
        return circ

    def loc_conv(self, qubit1, qubit2, circ, symbols):

        circ.append(cirq.rx(symbols[0]).on(qubit1))
        circ.append(cirq.ry(symbols[1]).on(qubit1))
        circ.append(cirq.rz(symbols[2]).on(qubit1)) 
        
        
        circ.append(cirq.rx(symbols[3]).on(qubit2))
        circ.append(cirq.ry(symbols[4]).on(qubit2))
        circ.append(cirq.rz(symbols[5]).on(qubit2))                
        
        return circ

    
    def qdcnn(self, qubits, circ, symbols, paras, nQconv):
        loc = qubits[:6]
        color = qubits[6:9]
        kernel = qubits[9:10]
        readout = qubits[10:12]
        circ.append(cirq.H.on_each(kernel))
        for i in range(nQconv):
            for j in range(len(color)):
                circ = self.loc_conv(qubits[3], qubits[0], circ, paras[54*i+18*j+0:54*i+18*j+6])
                circ = conv_layer(circ, qubits[3:6], qubits[0:3], color[j], kernel, readout[0], symbols[i*144+48*j:i*144+48*j+24])
                circ = self.loc_conv(qubits[4], qubits[1], circ, paras[54*i+18*j+6:54*i+18*j+12])
                circ = conv_layer(circ, qubits[4:6], qubits[1:3], readout[0], kernel, readout[1], symbols[i*144+48*j+24:i*144+48*j+48])   
                circ = self.loc_conv(qubits[5], qubits[2], circ, paras[54*i+18*j+12:54*i+18*j+18])
        
        return circ
    
    
    def call(self, inputs):
        inputs = self.flatten(inputs)
        batch_dim = tf.gather(tf.shape(inputs), 0)
        tiled_up_circuits = tf.repeat(self.circ, repeats=batch_dim)
        tiled_up_operators = tf.tile(self.operators, multiples=[batch_dim, 1])
        tiled_up_conv = tf.tile(self.conv, multiples=[batch_dim, 1])
        tiled_up_loc_p = tf.tile(self.loc_p, multiples=[batch_dim, 1])
        tiled_up_loc_u3 = tf.tile(self.loc_u3, multiples=[batch_dim, 1])

        joined_vars = tf.concat([tiled_up_loc_u3, inputs, tiled_up_loc_p, tiled_up_conv], axis=1)

        return self.executor(tiled_up_circuits, symbol_names=self.symbols, symbol_values=joined_vars, operators=tiled_up_operators)


class allEntEncodingPQC_u3(tf.keras.layers.Layer):
    def __init__(self, name="PQC"):
        super(allEntEncodingPQC_u3, self).__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        
        nElements = 9
        nQubits = 12

        qubits = cirq.GridQubit.rect(1, nQubits)
        observables = readout('allEntM', qubits)

        inputs = sympy.symbols(f'img:{64*nElements}')
        params_conv = sympy.symbols(f'conv:{144}')
        params_loc = sympy.symbols(f'loc:{81}')

        circ = cirq.Circuit()
        circ = self.encoding(qubits, circ, inputs)
        circ = self.qdcnn(qubits, circ, params_conv, params_loc, 1)

        self.conv = self.add_weight('conv', shape=(1, len(list(params_conv))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        
        self.loc_p = self.add_weight('loc_p', shape=(1, len(list(params_loc))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)

        self.symbols = [str(symb) for symb in list(inputs) + list(params_loc) + list(params_conv)]

        self.circ = tfq.convert_to_tensor([circ])
        self.operators = tfq.convert_to_tensor([observables])
        self.executor = expectation.Expectation(backend='noiseless', differentiator=None)
        
    def encoding(self, qubits, circ, inputs):
        loc = qubits[:6]
        target = qubits[6:9]
        circ.append(cirq.H.on_each(loc))
                
        for i in range(8):
            for j in range(8):
                row = [int(binary) for binary in format(i, '03b')]
                column = [int(binary) for binary in format(j, '03b')]
                ctrl_state = row + column
                for n in range(len(target)):
                    circ.append(cirq.rx(inputs[9*(8*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                    circ.append(cirq.ry(inputs[9*(8*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                    circ.append(cirq.rz(inputs[9*(8*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))

                circ.append(cirq.CZ(target[0], target[1]))
                circ.append(cirq.CZ(target[1], target[2]))
                circ.append(cirq.CZ(target[2], target[0]))

        return circ

    def loc_conv(self, qubit1, qubit2, qubit3, circ, symbols):

        circ.append(cirq.rx(symbols[0]).on(qubit1))
        circ.append(cirq.ry(symbols[1]).on(qubit1))
        circ.append(cirq.rz(symbols[2]).on(qubit1)) 
        
        
        circ.append(cirq.rx(symbols[3]).on(qubit2))
        circ.append(cirq.ry(symbols[4]).on(qubit2))
        circ.append(cirq.rz(symbols[5]).on(qubit2))    
        
        
        circ.append(cirq.rx(symbols[6]).on(qubit3))
        circ.append(cirq.ry(symbols[7]).on(qubit3))
        circ.append(cirq.rz(symbols[8]).on(qubit3))            
        
        return circ

        
    def qdcnn(self, qubits, circ, symbols, paras, nQconv):
        loc = qubits[:6]
        color = qubits[6:9]
        kernel = qubits[9:10]
        readout = qubits[10:12]
        circ.append(cirq.H.on_each(kernel))
        for i in range(nQconv):
            for j in range(len(color)):
                circ = self.loc_conv(qubits[3], qubits[0], kernel[0], circ, paras[81*i+27*j+0:81*i+27*j+9])
                circ = conv_layer(circ, qubits[3:6], qubits[0:3], color[j], kernel, readout[0], symbols[i*144+48*j:i*144+48*j+24])
                circ = self.loc_conv(qubits[4], qubits[1], kernel[0], circ, paras[81*i+27*j+9:81*i+27*j+18])
                circ = conv_layer(circ, qubits[4:6], qubits[1:3], readout[0], kernel, readout[1], symbols[i*144+48*j+24:i*144+48*j+48])   
                circ = self.loc_conv(qubits[5], qubits[2], kernel[0], circ, paras[81*i+27*j+18:81*i+27*j+27])
        
        return circ
        
    def call(self, inputs):
        inputs = self.flatten(inputs)
        batch_dim = tf.gather(tf.shape(inputs), 0)
        tiled_up_circuits = tf.repeat(self.circ, repeats=batch_dim)
        tiled_up_operators = tf.tile(self.operators, multiples=[batch_dim, 1])
        tiled_up_conv = tf.tile(self.conv, multiples=[batch_dim, 1])
        tiled_up_loc_p = tf.tile(self.loc_p, multiples=[batch_dim, 1])

        joined_vars = tf.concat([inputs, tiled_up_loc_p, tiled_up_conv], axis=1)

        return self.executor(tiled_up_circuits, symbol_names=self.symbols, symbol_values=joined_vars, operators=tiled_up_operators)


    
class allEntEncodingPQC_woLoc(tf.keras.layers.Layer):
    def __init__(self, name="PQC"):
        super(allEntEncodingPQC_woLoc, self).__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        
        nElements = 9
        nQubits = 12

        qubits = cirq.GridQubit.rect(1, nQubits)
        observables = readout('allEntM', qubits)

        inputs = sympy.symbols(f'img:{64*nElements}')
        params_conv = sympy.symbols(f'conv:{144}')

        circ = cirq.Circuit()
        circ = self.encoding(qubits, circ, inputs)
        circ = self.qdcnn(qubits, circ, params_conv, 1)

        self.conv = self.add_weight('conv', shape=(1, len(list(params_conv))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        

        self.symbols = [str(symb) for symb in list(inputs) + list(params_conv)]

        self.circ = tfq.convert_to_tensor([circ])
        self.operators = tfq.convert_to_tensor([observables])
        self.executor = expectation.Expectation(backend='noiseless', differentiator=None)
        
    def encoding(self, qubits, circ, inputs):
        loc = qubits[:6]
        target = qubits[6:9]
        circ.append(cirq.H.on_each(loc))
                
        for i in range(8):
            for j in range(8):
                row = [int(binary) for binary in format(i, '03b')]
                column = [int(binary) for binary in format(j, '03b')]
                ctrl_state = row + column
                for n in range(len(target)):
                    circ.append(cirq.rx(inputs[9*(8*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                    circ.append(cirq.ry(inputs[9*(8*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                    circ.append(cirq.rz(inputs[9*(8*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))

                circ.append(cirq.CZ(target[0], target[1]))
                circ.append(cirq.CZ(target[1], target[2]))
                circ.append(cirq.CZ(target[2], target[0]))

        return circ
        
    def qdcnn(self, qubits, circ, symbols, nQconv):
        loc = qubits[:6]
        color = qubits[6:9]
        kernel = qubits[9:10]
        readout = qubits[10:12]
        circ.append(cirq.H.on_each(kernel))
        for i in range(nQconv):
            for j in range(len(color)):
                circ = conv_layer(circ, qubits[3:6], qubits[0:3], color[j], kernel, readout[0], symbols[i*144+48*j:i*144+48*j+24])
                circ = conv_layer(circ, qubits[4:6], qubits[1:3], readout[0], kernel, readout[1], symbols[i*144+48*j+24:i*144+48*j+48])        
        
        return circ
        
    def call(self, inputs):
        inputs = self.flatten(inputs)
        batch_dim = tf.gather(tf.shape(inputs), 0)
        tiled_up_circuits = tf.repeat(self.circ, repeats=batch_dim)
        tiled_up_operators = tf.tile(self.operators, multiples=[batch_dim, 1])
        tiled_up_conv = tf.tile(self.conv, multiples=[batch_dim, 1])

        joined_vars = tf.concat([inputs, tiled_up_conv], axis=1)

        return self.executor(tiled_up_circuits, symbol_names=self.symbols, symbol_values=joined_vars, operators=tiled_up_operators)

    
    
class multiScaleEncodingPQC(tf.keras.layers.Layer):
    def __init__(self, name="PQC"):
        super(multiScaleEncodingPQC, self).__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        
        nElements = 9
        nQubits = 12

        qubits = cirq.GridQubit.rect(1, nQubits)
        observables = readout('allEntM', qubits)

        lInputs = sympy.symbols(f'imgl:{64*nElements}')
        mInputs = sympy.symbols(f'imgm:{64*nElements}')
        sInputs = sympy.symbols(f'imgs:{64*nElements}')

        params_conv1 = sympy.symbols(f'convl:{144}')
        params_conv2 = sympy.symbols(f'convm:{144}')
        params_conv3 = sympy.symbols(f'convs:{144}')
        
        params_loc1 = sympy.symbols(f'locl:{27}')
        params_loc2 = sympy.symbols(f'locm:{27}')
        params_loc3 = sympy.symbols(f'locs:{27}')


        circ = cirq.Circuit()
        circ = self.encoding(qubits, circ, lInputs)
        circ = self.qdcnn(qubits, circ, params_conv1, params_loc1, 1)
        circ = self.encoding(qubits, circ, mInputs)
        circ = self.qdcnn(qubits, circ, params_conv2, params_loc2, 1)
        circ = self.encoding(qubits, circ, sInputs)
        circ = self.qdcnn(qubits, circ, params_conv3, params_loc3, 1)

        self.conv1 = self.add_weight('conv1', shape=(1, len(list(params_conv1))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        self.conv2 = self.add_weight('conv2', shape=(1, len(list(params_conv2))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        self.conv3 = self.add_weight('conv3', shape=(1, len(list(params_conv3))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        
        
        self.loc1 = self.add_weight('loc1', shape=(1, len(list(params_loc1))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        self.loc2 = self.add_weight('loc2', shape=(1, len(list(params_loc2))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        self.loc3 = self.add_weight('loc3', shape=(1, len(list(params_loc3))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        
        
        self.symbols = [str(symb) for symb in 
                        list(lInputs) + list(params_conv1) + list(params_loc1) + 
                        list(mInputs) + list(params_conv2) + list(params_loc2) +
                        list(sInputs) + list(params_conv3) + list(params_loc3) ]

        self.circ = tfq.convert_to_tensor([circ])
        self.operators = tfq.convert_to_tensor([observables])
        self.executor = expectation.Expectation(backend='noiseless', differentiator=None)

    
    def encoding(self, qubits, circ, inputs):
        loc = qubits[:6]
        target = qubits[6:9]
        circ.append(cirq.H.on_each(loc))
        
        if len(inputs) == 64*9:
            for i in range(8):
                for j in range(8):
                    row = [int(binary) for binary in format(i, '03b')]
                    column = [int(binary) for binary in format(j, '03b')]
                    ctrl_state = row + column
                    for n in range(len(target)):
                        circ.append(cirq.rx(inputs[9*(8*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                        circ.append(cirq.ry(inputs[9*(8*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                        circ.append(cirq.rz(inputs[9*(8*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
        if len(inputs) == 16*9:    
            for i in range(4):
                for j in range(4):
                    row = [int(binary) for binary in format(i, '02b')]
                    column = [int(binary) for binary in format(j, '02b')]
                    ctrl_state = row + column
                    for n in range(len(target)):
                        circ.append(cirq.rx(inputs[9*(4*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[4], loc[2], loc[1], control_values=ctrl_state))
                        circ.append(cirq.ry(inputs[9*(4*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[4], loc[2], loc[1], control_values=ctrl_state))
                        circ.append(cirq.rz(inputs[9*(4*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[4], loc[2], loc[1], control_values=ctrl_state))
        if len(inputs) == 16*9:         
            for i in range(2):
                for j in range(2):
                    row = [int(binary) for binary in format(i, '01b')]
                    column = [int(binary) for binary in format(j, '01b')]
                    ctrl_state = row + column
                    for n in range(len(target)):
                        circ.append(cirq.rx(inputs[9*(2*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[2], control_values=ctrl_state))
                        circ.append(cirq.ry(inputs[9*(2*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[2], control_values=ctrl_state))
                        circ.append(cirq.rz(inputs[9*(2*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[2], control_values=ctrl_state))
        return circ

    def loc_conv(self, qubit1, qubit2, circ, symbols):
        circ.append(cirq.ry(symbols[0]).on(qubit1))
        circ.append(cirq.ry(symbols[1]).on(qubit2).controlled_by(qubit1, control_values=[0]))
        circ.append(cirq.ry(symbols[2]).on(qubit2).controlled_by(qubit1, control_values=[1]))
        return circ
    

    def qdcnn(self, qubits, circ, symbols, paras, nQconv):
        loc = qubits[:6]
        color = qubits[6:9]
        kernel = qubits[9:10]
        readout = qubits[10:12]
        circ.append(cirq.H.on_each(kernel))
        for i in range(nQconv):
            for j in range(len(color)):
                circ = self.loc_conv(qubits[3], qubits[0], circ, paras[27*i+9*j+0:27*i+9*j+3])
                circ = conv_layer(circ, qubits[3:6], qubits[0:3], color[j], kernel, readout[0], symbols[i*144+48*j:i*144+48*j+24])
                circ = self.loc_conv(qubits[4], qubits[1], circ, paras[27*i+9*j+3:27*i+9*j+6])
                circ = conv_layer(circ, qubits[4:6], qubits[1:3], readout[0], kernel, readout[1], symbols[i*144+48*j+24:i*144+48*j+48])        
                circ = self.loc_conv(qubits[5], qubits[2], circ, paras[27*i+9*j+6:27*i+9*j+9])
        
        return circ
    
    def call(self, inputs):
        lInputs, mInputs, sInputs = inputs
        
        lInputs = self.flatten(lInputs)
        mInputs = self.flatten(mInputs)
        sInputs = self.flatten(sInputs)
                
        batch_dim = tf.gather(tf.shape(mInputs), 0)
        tiled_up_circuits = tf.repeat(self.circ, repeats=batch_dim)
        tiled_up_operators = tf.tile(self.operators, multiples=[batch_dim, 1])
        tiled_up_conv1 = tf.tile(self.conv1, multiples=[batch_dim, 1])
        tiled_up_conv2 = tf.tile(self.conv2, multiples=[batch_dim, 1])
        tiled_up_conv3 = tf.tile(self.conv3, multiples=[batch_dim, 1])
        
        tiled_up_loc1 = tf.tile(self.loc1, multiples=[batch_dim, 1])
        tiled_up_loc2 = tf.tile(self.loc2, multiples=[batch_dim, 1])
        tiled_up_loc3 = tf.tile(self.loc3, multiples=[batch_dim, 1])

        joined_vars = tf.concat([lInputs, tiled_up_conv1, tiled_up_loc1,
                                 mInputs, tiled_up_conv2, tiled_up_loc2,
                                 sInputs, tiled_up_conv3, tiled_up_loc3], axis=1)

        return self.executor(tiled_up_circuits, symbol_names=self.symbols, symbol_values=joined_vars, operators=tiled_up_operators)
    
    
    
    
    

    
class multiScaleEncodingPQC_woLoc(tf.keras.layers.Layer):
    def __init__(self, name="PQC"):
        super(multiScaleEncodingPQC_woLoc, self).__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        
        nElements = 9
        nQubits = 12

        qubits = cirq.GridQubit.rect(1, nQubits)
        observables = readout('allEntM', qubits)

        lInputs = sympy.symbols(f'imgl:{64*nElements}')
        mInputs = sympy.symbols(f'imgm:{64*nElements}')
        sInputs = sympy.symbols(f'imgs:{64*nElements}')

        params_conv1 = sympy.symbols(f'convl:{144}')
        params_conv2 = sympy.symbols(f'convm:{144}')
        params_conv3 = sympy.symbols(f'convs:{144}')
        

        circ = cirq.Circuit()
        circ = self.encoding(qubits, circ, lInputs)
        circ = self.qdcnn(qubits, circ, params_conv1, 1)
        circ = self.encoding(qubits, circ, mInputs)
        circ = self.qdcnn(qubits, circ, params_conv2, 1)
        circ = self.encoding(qubits, circ, sInputs)
        circ = self.qdcnn(qubits, circ, params_conv3, 1)

        self.conv1 = self.add_weight('conv1', shape=(1, len(list(params_conv1))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        self.conv2 = self.add_weight('conv2', shape=(1, len(list(params_conv2))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        self.conv3 = self.add_weight('conv3', shape=(1, len(list(params_conv3))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        
    
        
        
        self.symbols = [str(symb) for symb in 
                        list(lInputs) + list(params_conv1) + 
                        list(mInputs) + list(params_conv2) +
                        list(sInputs) + list(params_conv3)]

        self.circ = tfq.convert_to_tensor([circ])
        self.operators = tfq.convert_to_tensor([observables])
        self.executor = expectation.Expectation(backend='noiseless', differentiator=None)

    
    def encoding(self, qubits, circ, inputs):
        loc = qubits[:6]
        target = qubits[6:9]
        circ.append(cirq.H.on_each(loc))
        
        if len(inputs) == 64*9:
            for i in range(8):
                for j in range(8):
                    row = [int(binary) for binary in format(i, '03b')]
                    column = [int(binary) for binary in format(j, '03b')]
                    ctrl_state = row + column
                    for n in range(len(target)):
                        circ.append(cirq.rx(inputs[9*(8*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                        circ.append(cirq.ry(inputs[9*(8*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                        circ.append(cirq.rz(inputs[9*(8*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
        if len(inputs) == 16*9:    
            for i in range(4):
                for j in range(4):
                    row = [int(binary) for binary in format(i, '02b')]
                    column = [int(binary) for binary in format(j, '02b')]
                    ctrl_state = row + column
                    for n in range(len(target)):
                        circ.append(cirq.rx(inputs[9*(4*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[4], loc[2], loc[1], control_values=ctrl_state))
                        circ.append(cirq.ry(inputs[9*(4*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[4], loc[2], loc[1], control_values=ctrl_state))
                        circ.append(cirq.rz(inputs[9*(4*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[4], loc[2], loc[1], control_values=ctrl_state))
        if len(inputs) == 16*9:         
            for i in range(2):
                for j in range(2):
                    row = [int(binary) for binary in format(i, '01b')]
                    column = [int(binary) for binary in format(j, '01b')]
                    ctrl_state = row + column
                    for n in range(len(target)):
                        circ.append(cirq.rx(inputs[9*(2*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[2], control_values=ctrl_state))
                        circ.append(cirq.ry(inputs[9*(2*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[2], control_values=ctrl_state))
                        circ.append(cirq.rz(inputs[9*(2*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[2], control_values=ctrl_state))
        return circ

    def qdcnn(self, qubits, circ, symbols, nQconv):
        loc = qubits[:6]
        color = qubits[6:9]
        kernel = qubits[9:10]
        readout = qubits[10:12]
        circ.append(cirq.H.on_each(kernel))
        for i in range(nQconv):
            for j in range(len(color)):
                circ = conv_layer(circ, qubits[3:6], qubits[0:3], color[j], kernel, readout[0], symbols[i*144+48*j:i*144+48*j+24])
                circ = conv_layer(circ, qubits[4:6], qubits[1:3], readout[0], kernel, readout[1], symbols[i*144+48*j+24:i*144+48*j+48])        
        
        return circ
    
    def call(self, inputs):
        lInputs, mInputs, sInputs = inputs
        
        lInputs = self.flatten(lInputs)
        mInputs = self.flatten(mInputs)
        sInputs = self.flatten(sInputs)
                
        batch_dim = tf.gather(tf.shape(mInputs), 0)
        tiled_up_circuits = tf.repeat(self.circ, repeats=batch_dim)
        tiled_up_operators = tf.tile(self.operators, multiples=[batch_dim, 1])
        tiled_up_conv1 = tf.tile(self.conv1, multiples=[batch_dim, 1])
        tiled_up_conv2 = tf.tile(self.conv2, multiples=[batch_dim, 1])
        tiled_up_conv3 = tf.tile(self.conv3, multiples=[batch_dim, 1])
        

        joined_vars = tf.concat([lInputs, tiled_up_conv1,
                                 mInputs, tiled_up_conv2,
                                 sInputs, tiled_up_conv3], axis=1)

        return self.executor(tiled_up_circuits, symbol_names=self.symbols, symbol_values=joined_vars, operators=tiled_up_operators)

    
class multiScaleV2EncodingPQC(tf.keras.layers.Layer):
    def __init__(self, name="PQC"):
        super(multiScaleV2EncodingPQC, self).__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        
        nElements = 9
        nQubits = 12

        qubits = cirq.GridQubit.rect(1, nQubits)
        observables = readout('allEntM', qubits)

        lInputs = sympy.symbols(f'imgl:{64*nElements}')
        mInputs = sympy.symbols(f'imgm:{16*nElements}')
        sInputs = sympy.symbols(f'imgs:{4*nElements}')

        params_conv1 = sympy.symbols(f'convl:{144}')
        params_conv2 = sympy.symbols(f'convm:{144}')
        params_conv3 = sympy.symbols(f'convs:{144}')
        
        params_loc1 = sympy.symbols(f'locl:{27}')
        params_loc2 = sympy.symbols(f'locm:{27}')
        params_loc3 = sympy.symbols(f'locs:{27}')


        circ = cirq.Circuit()
        circ = self.encoding(qubits, circ, lInputs)
        circ = self.qdcnn(qubits, circ, params_conv1, params_loc1, 1)
        circ = self.encoding(qubits, circ, mInputs)
        circ = self.qdcnn(qubits, circ, params_conv2, params_loc2, 1)
        circ = self.encoding(qubits, circ, sInputs)
        circ = self.qdcnn(qubits, circ, params_conv3, params_loc3, 1)

        self.conv1 = self.add_weight('conv1', shape=(1, len(list(params_conv1))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        self.conv2 = self.add_weight('conv2', shape=(1, len(list(params_conv2))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        self.conv3 = self.add_weight('conv3', shape=(1, len(list(params_conv3))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        
        
        self.loc1 = self.add_weight('loc1', shape=(1, len(list(params_loc1))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        self.loc2 = self.add_weight('loc2', shape=(1, len(list(params_loc2))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        self.loc3 = self.add_weight('loc3', shape=(1, len(list(params_loc3))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        
        
        self.symbols = [str(symb) for symb in 
                        list(lInputs) + list(params_conv1) + list(params_loc1) + 
                        list(mInputs) + list(params_conv2) + list(params_loc2) +
                        list(sInputs) + list(params_conv3) + list(params_loc3) ]

        self.circ = tfq.convert_to_tensor([circ])
        self.operators = tfq.convert_to_tensor([observables])
        self.executor = expectation.Expectation(backend='noiseless', differentiator=None)

    
    def encoding(self, qubits, circ, inputs):
        loc = qubits[:6]
        target = qubits[6:9]
        circ.append(cirq.H.on_each(loc))
        
        if len(inputs) == 64*9:
            for i in range(8):
                for j in range(8):
                    row = [int(binary) for binary in format(i, '03b')]
                    column = [int(binary) for binary in format(j, '03b')]
                    ctrl_state = row + column
                    for n in range(len(target)):
                        circ.append(cirq.rx(inputs[9*(8*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                        circ.append(cirq.ry(inputs[9*(8*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                        circ.append(cirq.rz(inputs[9*(8*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
        if len(inputs) == 16*9:    
            for i in range(4):
                for j in range(4):
                    row = [int(binary) for binary in format(i, '02b')]
                    column = [int(binary) for binary in format(j, '02b')]
                    ctrl_state = row + column
                    for n in range(len(target)):
                        circ.append(cirq.rx(inputs[9*(4*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[4], loc[2], loc[1], control_values=ctrl_state))
                        circ.append(cirq.ry(inputs[9*(4*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[4], loc[2], loc[1], control_values=ctrl_state))
                        circ.append(cirq.rz(inputs[9*(4*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[4], loc[2], loc[1], control_values=ctrl_state))
        if len(inputs) == 16*9:         
            for i in range(2):
                for j in range(2):
                    row = [int(binary) for binary in format(i, '01b')]
                    column = [int(binary) for binary in format(j, '01b')]
                    ctrl_state = row + column
                    for n in range(len(target)):
                        circ.append(cirq.rx(inputs[9*(2*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[2], control_values=ctrl_state))
                        circ.append(cirq.ry(inputs[9*(2*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[2], control_values=ctrl_state))
                        circ.append(cirq.rz(inputs[9*(2*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[2], control_values=ctrl_state))
        return circ

    def loc_conv(self, qubit1, qubit2, circ, symbols):
        circ.append(cirq.ry(symbols[0]).on(qubit1))
        circ.append(cirq.ry(symbols[1]).on(qubit2).controlled_by(qubit1, control_values=[0]))
        circ.append(cirq.ry(symbols[2]).on(qubit2).controlled_by(qubit1, control_values=[1]))
        return circ
    

    def qdcnn(self, qubits, circ, symbols, paras, nQconv):
        loc = qubits[:6]
        color = qubits[6:9]
        kernel = qubits[9:10]
        readout = qubits[10:12]
        circ.append(cirq.H.on_each(kernel))
        for i in range(nQconv):
            for j in range(len(color)):
                circ = self.loc_conv(qubits[3], qubits[0], circ, paras[27*i+9*j+0:27*i+9*j+3])
                circ = conv_layer(circ, qubits[3:6], qubits[0:3], color[j], kernel, readout[0], symbols[i*144+48*j:i*144+48*j+24])
                circ = self.loc_conv(qubits[4], qubits[1], circ, paras[27*i+9*j+3:27*i+9*j+6])
                circ = conv_layer(circ, qubits[4:6], qubits[1:3], readout[0], kernel, readout[1], symbols[i*144+48*j+24:i*144+48*j+48])        
                circ = self.loc_conv(qubits[5], qubits[2], circ, paras[27*i+9*j+6:27*i+9*j+9])
        
        return circ
    
    def call(self, inputs):
        lInputs, mInputs, sInputs = inputs
        
        lInputs = self.flatten(lInputs)
        mInputs = self.flatten(mInputs)
        sInputs = self.flatten(sInputs)
                
        batch_dim = tf.gather(tf.shape(mInputs), 0)
        tiled_up_circuits = tf.repeat(self.circ, repeats=batch_dim)
        tiled_up_operators = tf.tile(self.operators, multiples=[batch_dim, 1])
        tiled_up_conv1 = tf.tile(self.conv1, multiples=[batch_dim, 1])
        tiled_up_conv2 = tf.tile(self.conv2, multiples=[batch_dim, 1])
        tiled_up_conv3 = tf.tile(self.conv3, multiples=[batch_dim, 1])
        
        tiled_up_loc1 = tf.tile(self.loc1, multiples=[batch_dim, 1])
        tiled_up_loc2 = tf.tile(self.loc2, multiples=[batch_dim, 1])
        tiled_up_loc3 = tf.tile(self.loc3, multiples=[batch_dim, 1])

        joined_vars = tf.concat([lInputs, tiled_up_conv1, tiled_up_loc1,
                                 mInputs, tiled_up_conv2, tiled_up_loc2,
                                 sInputs, tiled_up_conv3, tiled_up_loc3], axis=1)

        return self.executor(tiled_up_circuits, symbol_names=self.symbols, symbol_values=joined_vars, operators=tiled_up_operators)
    
    
    
class multiScaleV2EncodingPQC_woLoc(tf.keras.layers.Layer):
    def __init__(self, name="PQC"):
        super(multiScaleV2EncodingPQC_woLoc, self).__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        
        nElements = 9
        nQubits = 12

        qubits = cirq.GridQubit.rect(1, nQubits)
        observables = readout('allEntM', qubits)

        lInputs = sympy.symbols(f'imgl:{64*nElements}')
        mInputs = sympy.symbols(f'imgm:{16*nElements}')
        sInputs = sympy.symbols(f'imgs:{4*nElements}')

        params_conv1 = sympy.symbols(f'convl:{144}')
        params_conv2 = sympy.symbols(f'convm:{144}')
        params_conv3 = sympy.symbols(f'convs:{144}')
        

        circ = cirq.Circuit()
        circ = self.encoding(qubits, circ, lInputs)
        circ = self.qdcnn(qubits, circ, params_conv1, 1)
        circ = self.encoding(qubits, circ, mInputs)
        circ = self.qdcnn(qubits, circ, params_conv2, 1)
        circ = self.encoding(qubits, circ, sInputs)
        circ = self.qdcnn(qubits, circ, params_conv3, 1)

        self.conv1 = self.add_weight('conv1', shape=(1, len(list(params_conv1))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        self.conv2 = self.add_weight('conv2', shape=(1, len(list(params_conv2))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        self.conv3 = self.add_weight('conv3', shape=(1, len(list(params_conv3))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        
        
        
        self.symbols = [str(symb) for symb in 
                        list(lInputs) + list(params_conv1) + 
                        list(mInputs) + list(params_conv2) + 
                        list(sInputs) + list(params_conv3)]

        self.circ = tfq.convert_to_tensor([circ])
        self.operators = tfq.convert_to_tensor([observables])
        self.executor = expectation.Expectation(backend='noiseless', differentiator=None)

    
    def encoding(self, qubits, circ, inputs):
        loc = qubits[:6]
        target = qubits[6:9]
        circ.append(cirq.H.on_each(loc))
        
        if len(inputs) == 64*9:
            for i in range(8):
                for j in range(8):
                    row = [int(binary) for binary in format(i, '03b')]
                    column = [int(binary) for binary in format(j, '03b')]
                    ctrl_state = row + column
                    for n in range(len(target)):
                        circ.append(cirq.rx(inputs[9*(8*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                        circ.append(cirq.ry(inputs[9*(8*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                        circ.append(cirq.rz(inputs[9*(8*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
        if len(inputs) == 16*9:    
            for i in range(4):
                for j in range(4):
                    row = [int(binary) for binary in format(i, '02b')]
                    column = [int(binary) for binary in format(j, '02b')]
                    ctrl_state = row + column
                    for n in range(len(target)):
                        circ.append(cirq.rx(inputs[9*(4*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[4], loc[2], loc[1], control_values=ctrl_state))
                        circ.append(cirq.ry(inputs[9*(4*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[4], loc[2], loc[1], control_values=ctrl_state))
                        circ.append(cirq.rz(inputs[9*(4*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[4], loc[2], loc[1], control_values=ctrl_state))
        if len(inputs) == 16*9:         
            for i in range(2):
                for j in range(2):
                    row = [int(binary) for binary in format(i, '01b')]
                    column = [int(binary) for binary in format(j, '01b')]
                    ctrl_state = row + column
                    for n in range(len(target)):
                        circ.append(cirq.rx(inputs[9*(2*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[2], control_values=ctrl_state))
                        circ.append(cirq.ry(inputs[9*(2*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[2], control_values=ctrl_state))
                        circ.append(cirq.rz(inputs[9*(2*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[2], control_values=ctrl_state))
        return circ



    def qdcnn(self, qubits, circ, symbols, nQconv):
        loc = qubits[:6]
        color = qubits[6:9]
        kernel = qubits[9:10]
        readout = qubits[10:12]
        circ.append(cirq.H.on_each(kernel))
        for i in range(nQconv):
            for j in range(len(color)):
                circ = conv_layer(circ, qubits[3:6], qubits[0:3], color[j], kernel, readout[0], symbols[i*144+48*j:i*144+48*j+24])
                circ = conv_layer(circ, qubits[4:6], qubits[1:3], readout[0], kernel, readout[1], symbols[i*144+48*j+24:i*144+48*j+48])        
        
        return circ
    
    def call(self, inputs):
        lInputs, mInputs, sInputs = inputs
        
        lInputs = self.flatten(lInputs)
        mInputs = self.flatten(mInputs)
        sInputs = self.flatten(sInputs)
                
        batch_dim = tf.gather(tf.shape(mInputs), 0)
        tiled_up_circuits = tf.repeat(self.circ, repeats=batch_dim)
        tiled_up_operators = tf.tile(self.operators, multiples=[batch_dim, 1])
        tiled_up_conv1 = tf.tile(self.conv1, multiples=[batch_dim, 1])
        tiled_up_conv2 = tf.tile(self.conv2, multiples=[batch_dim, 1])
        tiled_up_conv3 = tf.tile(self.conv3, multiples=[batch_dim, 1])


        joined_vars = tf.concat([lInputs, tiled_up_conv1,
                                 mInputs, tiled_up_conv2,
                                 sInputs, tiled_up_conv3], axis=1)

        return self.executor(tiled_up_circuits, symbol_names=self.symbols, symbol_values=joined_vars, operators=tiled_up_operators)
    
    
class multiScaleV3EncodingPQC_woLoc(tf.keras.layers.Layer):
    def __init__(self, name="PQC"):
        super(multiScaleV3EncodingPQC_woLoc, self).__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        
        nElements = 9
        nQubits = 12

        qubits = cirq.GridQubit.rect(1, nQubits)
        observables = readout('allEntM', qubits)

        lInputs = sympy.symbols(f'imgl:{64*nElements}')
        mInputs = sympy.symbols(f'imgm:{64*nElements}')
        sInputs = sympy.symbols(f'imgs:{64*nElements}')

        params_conv1 = sympy.symbols(f'convl:{144}')
        params_conv2 = sympy.symbols(f'convm:{144}')
        params_conv3 = sympy.symbols(f'convs:{144}')
        

        circ = cirq.Circuit()
        circ = self.encoding(qubits, circ, lInputs)
        circ = self.qdcnn(qubits, circ, params_conv1, 1)
        circ = self.encoding(qubits, circ, mInputs)
        circ = self.qdcnn(qubits, circ, params_conv2, 1)
        circ = self.encoding(qubits, circ, sInputs)
        circ = self.qdcnn(qubits, circ, params_conv3, 1)

        self.conv1 = self.add_weight('conv1', shape=(1, len(list(params_conv1))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        self.conv2 = self.add_weight('conv2', shape=(1, len(list(params_conv2))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        self.conv3 = self.add_weight('conv3', shape=(1, len(list(params_conv3))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)
        
        
        
        self.symbols = [str(symb) for symb in 
                        list(lInputs) + list(params_conv1) + 
                        list(mInputs) + list(params_conv2) + 
                        list(sInputs) + list(params_conv3)]

        self.circ = tfq.convert_to_tensor([circ])
        self.operators = tfq.convert_to_tensor([observables])
        self.executor = expectation.Expectation(backend='noiseless', differentiator=None)

    
    def encoding(self, qubits, circ, inputs):
        loc = qubits[:6]
        target = qubits[6:9]
        circ.append(cirq.H.on_each(loc))
        
        if len(inputs) == 64*9:
            for i in range(8):
                for j in range(8):
                    row = [int(binary) for binary in format(i, '03b')]
                    column = [int(binary) for binary in format(j, '03b')]
                    ctrl_state = row + column
                    for n in range(len(target)):
                        circ.append(cirq.rx(inputs[9*(8*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                        circ.append(cirq.ry(inputs[9*(8*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                        circ.append(cirq.rz(inputs[9*(8*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
        if len(inputs) == 16*9:    
            for i in range(4):
                for j in range(4):
                    row = [int(binary) for binary in format(i, '02b')]
                    column = [int(binary) for binary in format(j, '02b')]
                    ctrl_state = row + column
                    for n in range(len(target)):
                        circ.append(cirq.rx(inputs[9*(4*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[4], loc[2], loc[1], control_values=ctrl_state))
                        circ.append(cirq.ry(inputs[9*(4*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[4], loc[2], loc[1], control_values=ctrl_state))
                        circ.append(cirq.rz(inputs[9*(4*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[4], loc[2], loc[1], control_values=ctrl_state))
        if len(inputs) == 16*9:         
            for i in range(2):
                for j in range(2):
                    row = [int(binary) for binary in format(i, '01b')]
                    column = [int(binary) for binary in format(j, '01b')]
                    ctrl_state = row + column
                    for n in range(len(target)):
                        circ.append(cirq.rx(inputs[9*(2*i+j)+3*n+0]).on(target[n]).controlled_by(loc[5], loc[2], control_values=ctrl_state))
                        circ.append(cirq.ry(inputs[9*(2*i+j)+3*n+1]).on(target[n]).controlled_by(loc[5], loc[2], control_values=ctrl_state))
                        circ.append(cirq.rz(inputs[9*(2*i+j)+3*n+2]).on(target[n]).controlled_by(loc[5], loc[2], control_values=ctrl_state))
        return circ



    def qdcnn(self, qubits, circ, symbols, nQconv):
        loc = qubits[:6]
        color = qubits[6:9]
        kernel = qubits[9:10]
        readout = qubits[10:12]
        circ.append(cirq.H.on_each(kernel))
        for i in range(nQconv):
            for j in range(len(color)):
                circ = conv_layer(circ, qubits[3:6], qubits[0:3], color[j], kernel, readout[0], symbols[i*144+48*j:i*144+48*j+24])
                circ = conv_layer(circ, qubits[4:6], qubits[1:3], readout[0], kernel, readout[1], symbols[i*144+48*j+24:i*144+48*j+48])        
        
        return circ
    
    def call(self, inputs):
        lInputs, mInputs, sInputs = inputs
        
        lInputs = self.flatten(lInputs)
        mInputs = self.flatten(mInputs)
        sInputs = self.flatten(sInputs)
                
        batch_dim = tf.gather(tf.shape(mInputs), 0)
        tiled_up_circuits = tf.repeat(self.circ, repeats=batch_dim)
        tiled_up_operators = tf.tile(self.operators, multiples=[batch_dim, 1])
        tiled_up_conv1 = tf.tile(self.conv1, multiples=[batch_dim, 1])
        tiled_up_conv2 = tf.tile(self.conv2, multiples=[batch_dim, 1])
        tiled_up_conv3 = tf.tile(self.conv3, multiples=[batch_dim, 1])


        joined_vars = tf.concat([lInputs, tiled_up_conv1,
                                 mInputs, tiled_up_conv2,
                                 sInputs, tiled_up_conv3], axis=1)

        return self.executor(tiled_up_circuits, symbol_names=self.symbols, symbol_values=joined_vars, operators=tiled_up_operators)
    
    
    
    

class multiScaleV4EncodingPQC_woLoc(tf.keras.layers.Layer):
    def __init__(self, name="PQC"):
        super(multiScaleV4EncodingPQC_woLoc, self).__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        
        nElements = 9
        nQubits = 10

        qubits = cirq.GridQubit.rect(1, nQubits)
        observables = self.measure(qubits)

        lInputs = sympy.symbols(f'imgl:{64*nElements}')
        mInputs = sympy.symbols(f'imgm:{64*nElements}')
        sInputs = sympy.symbols(f'imgs:{64*nElements}')

        params_conv = sympy.symbols(f'conv:{432}')


        circ = cirq.Circuit()
        circ = self.encoding(qubits, circ, lInputs, mInputs, sInputs, params_conv, nElements)

        self.conv = self.add_weight('conv', shape=(1, len(list(params_conv))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)        
        
        self.symbols = [str(symb) for symb in 
                        list(lInputs) + list(mInputs) + list(sInputs) + list(params_conv)]

        self.circ = tfq.convert_to_tensor([circ])
        self.operators = tfq.convert_to_tensor([observables])
        self.executor = expectation.Expectation(backend='noiseless', differentiator=None)
        
    def measure(self, qubits):
        output, imgs, kernels, channels = [], [], [], []
        loc1, loc2, readout, kernel = qubits[2], qubits[5], qubits[9], qubits[7]
   
        imgs.append((1 + cirq.X(loc1)) * (1 + cirq.X(loc2)) * (1 - cirq.X(readout)))
        imgs.append((1 - cirq.X(loc1)) * (1 + cirq.X(loc2)) * (1 - cirq.X(readout)))
        imgs.append((1 + cirq.X(loc1)) * (1 - cirq.X(loc2)) * (1 - cirq.X(readout)))
        imgs.append((1 - cirq.X(loc1)) * (1 - cirq.X(loc2)) * (1 - cirq.X(readout)))

        kernels.append((1 + cirq.X(kernel)))
        kernels.append((1 - cirq.X(kernel)))

        for img in imgs:
            for kernel in kernels:
                output.append(img*kernel)        
        return output
    
    def encoding(self, qubits, circ, lInputs, mInputs, sInputs, symbols, nElements):
        loc = qubits[:6]
        target = qubits[6]
        kernel = qubits[7:8]
        readout = qubits[8:10]

        circ.append(cirq.H.on_each(loc))
        circ.append(cirq.H.on_each(kernel))
        
        for n in range(nElements):        
            for i in range(8):
                for j in range(8):
                    row = [int(binary) for binary in format(i, '03b')]
                    column = [int(binary) for binary in format(j, '03b')]
                    ctrl_state = row + column
                    
                    circ.append(cirq.rx(lInputs[9*(8*i+j)+n]).on(target).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                    circ.append(cirq.ry(mInputs[9*(8*i+j)+n]).on(target).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                    circ.append(cirq.rz(sInputs[9*(8*i+j)+n]).on(target).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
            
            circ = conv_layer(circ, qubits[3:6], qubits[0:3], target, kernel, readout[0], symbols[n*48:n*48+24])
            circ = conv_layer(circ, qubits[4:6], qubits[1:3], readout[0], kernel, readout[1], symbols[n*48+24:n*48+48])        
        return circ

    def call(self, inputs):
        lInputs, mInputs, sInputs = inputs
        
        lInputs = self.flatten(lInputs)
        mInputs = self.flatten(mInputs)
        sInputs = self.flatten(sInputs)
                
        batch_dim = tf.gather(tf.shape(mInputs), 0)
        tiled_up_circuits = tf.repeat(self.circ, repeats=batch_dim)
        tiled_up_operators = tf.tile(self.operators, multiples=[batch_dim, 1])
        tiled_up_conv = tf.tile(self.conv, multiples=[batch_dim, 1])


        joined_vars = tf.concat([lInputs, mInputs, sInputs, tiled_up_conv], axis=1)

        return self.executor(tiled_up_circuits, symbol_names=self.symbols, symbol_values=joined_vars, operators=tiled_up_operators)
    
    
    
class multiScaleV5EncodingPQC_woLoc(tf.keras.layers.Layer):
    def __init__(self, name="PQC"):
        super(multiScaleV5EncodingPQC_woLoc, self).__init__(name=name)
        self.flatten = tf.keras.layers.Flatten()
        
        nElements = 9
        nQubits = 12

        qubits = cirq.GridQubit.rect(1, nQubits)
        observables = self.measure(qubits)

        lInputs = sympy.symbols(f'imgl:{64*nElements}')
        mInputs = sympy.symbols(f'imgm:{64*nElements}')
        sInputs = sympy.symbols(f'imgs:{64*nElements}')

        params_conv = sympy.symbols(f'conv:{432}')


        circ = cirq.Circuit()
        circ = self.encoding(qubits, circ, lInputs, mInputs, sInputs, params_conv, nElements)

        self.conv = self.add_weight('conv', shape=(1, len(list(params_conv))),
                                    initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi), dtype=tf.float32,
                                    trainable=True)        
        
        self.symbols = [str(symb) for symb in 
                        list(lInputs) + list(mInputs) + list(sInputs) + list(params_conv)]

        self.circ = tfq.convert_to_tensor([circ])
        self.operators = tfq.convert_to_tensor([observables])
        self.executor = expectation.Expectation(backend='noiseless', differentiator=None)
        
    def measure(self, qubits):
        output, imgs, kernels, channels = [], [], [], []
        loc1, loc2, readout, kernel, entangle1, entangle2, entangle3 = qubits[2], qubits[5], qubits[11], qubits[9], qubits[6], qubits[7], qubits[8]

       
        imgs.append((1 + cirq.X(loc1)) * (1 + cirq.X(loc2)) * (1 - cirq.X(readout)))
        imgs.append((1 - cirq.X(loc1)) * (1 + cirq.X(loc2)) * (1 - cirq.X(readout)))
        imgs.append((1 + cirq.X(loc1)) * (1 - cirq.X(loc2)) * (1 - cirq.X(readout)))
        imgs.append((1 - cirq.X(loc1)) * (1 - cirq.X(loc2)) * (1 - cirq.X(readout)))

        kernels.append((1 + cirq.X(kernel)))
        kernels.append((1 - cirq.X(kernel)))

        channels.append((1 - cirq.X(entangle1)) * (1 - cirq.X(entangle2)) * (1 - cirq.X(entangle3)))
        channels.append((1 + cirq.X(entangle1)) * (1 - cirq.X(entangle2)) * (1 - cirq.X(entangle3)))
        channels.append((1 - cirq.X(entangle1)) * (1 + cirq.X(entangle2)) * (1 - cirq.X(entangle3)))
        channels.append((1 + cirq.X(entangle1)) * (1 + cirq.X(entangle2)) * (1 - cirq.X(entangle3)))
        channels.append((1 - cirq.X(entangle1)) * (1 - cirq.X(entangle2)) * (1 + cirq.X(entangle3)))
        channels.append((1 + cirq.X(entangle1)) * (1 - cirq.X(entangle2)) * (1 + cirq.X(entangle3)))
        channels.append((1 - cirq.X(entangle1)) * (1 + cirq.X(entangle2)) * (1 + cirq.X(entangle3)))
        channels.append((1 + cirq.X(entangle1)) * (1 + cirq.X(entangle2)) * (1 + cirq.X(entangle3)))

        for img in imgs:
            for kernel in kernels:
                for channel in channels:
                    output.append(img*kernel*channel)        
        return output

    
    def encoding(self, qubits, circ, lInputs, mInputs, sInputs, symbols, nElements):
        loc = qubits[:6]
        target = qubits[6:9]
        kernel = qubits[9:10]
        readout = qubits[10:12]

        circ.append(cirq.H.on_each(loc))
        circ.append(cirq.H.on_each(kernel))
        
        for n in range(int(nElements/3)):        
            for i in range(8):
                for j in range(8):
                    row = [int(binary) for binary in format(i, '03b')]
                    column = [int(binary) for binary in format(j, '03b')]
                    ctrl_state = row + column
                    
                    for k in range(len(target)):
                        circ.append(cirq.rx(lInputs[9*(8*i+j)+3*n+k]).on(target[k]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                        circ.append(cirq.ry(mInputs[9*(8*i+j)+3*n+k]).on(target[k]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                        circ.append(cirq.rz(sInputs[9*(8*i+j)+3*n+k]).on(target[k]).controlled_by(loc[5], loc[4], loc[3], loc[2], loc[1], loc[0], control_values=ctrl_state))
                        
            
            for x in range(len(target)):
                circ = conv_layer(circ, qubits[3:6], qubits[0:3], target[x], kernel, readout[0], symbols[n*144+48*x:n*144+48*x+24])
                circ = conv_layer(circ, qubits[4:6], qubits[1:3], readout[0], kernel, readout[1], symbols[n*144+48*x+24:n*144+48*x+48])              

        return circ

    def call(self, inputs):
        lInputs, mInputs, sInputs = inputs
        
        lInputs = self.flatten(lInputs)
        mInputs = self.flatten(mInputs)
        sInputs = self.flatten(sInputs)
                
        batch_dim = tf.gather(tf.shape(mInputs), 0)
        tiled_up_circuits = tf.repeat(self.circ, repeats=batch_dim)
        tiled_up_operators = tf.tile(self.operators, multiples=[batch_dim, 1])
        tiled_up_conv = tf.tile(self.conv, multiples=[batch_dim, 1])


        joined_vars = tf.concat([lInputs, mInputs, sInputs, tiled_up_conv], axis=1)

        return self.executor(tiled_up_circuits, symbol_names=self.symbols, symbol_values=joined_vars, operators=tiled_up_operators)
    
    
    
        