import h5py
import collections
import tensorflow as tf
import numpy as np

notop_weights_param_num = 234
weights_param_num = 236

def load_attributes_from_hdf5_group(group, name):
        if name in group.attrs:
                data = [n.decode('utf8') for n in group.attrs[name]]
        return data


# 1 open hdf5 file
# 2 get info
# 3 get layer names
# 4 get filtered_layer_names (which layer has weights)
# 5 get group by layer name
# 6 load layer's weight names from group by key 'weight_names'
# 7 load layer's weight value from group by weight names
# 8 fill all weights info into weights info dict

class WeightInfo():
        def __init__(self, name, shape, value):
                self.name = name
                self.shape = shape
                self.value = value
        
def parse_weights(weights_file = 'Xception_finetune_3.h5'):
        weights_dict = {}
        weights_dict = collections.OrderedDict()
        
        f = h5py.File(weights_file,'r')
#        print("   ", f.attrs['keras_version'].decode('utf8'))
#        print("   ", f.attrs['backend'].decode('utf8'))
        layer_names = load_attributes_from_hdf5_group(f, 'layer_names')

        filtered_layer_names = []
        for layer_name in layer_names:
                g = f[layer_name]
                weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
                if weight_names:
                        filtered_layer_names.append(layer_name)

        layer_names = filtered_layer_names
        for layer_name in layer_names:
                g = f[layer_name]
                weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
                weight_values = [g[weight_name] for weight_name in weight_names]

                for i in range(len(weight_names)):
                        weights_info = WeightInfo(name = weight_names[i], shape = weight_values[i].shape, value = weight_values[i][:])
                        weights_dict[weight_names[i]] = weights_info

        return weights_dict

def load_weights_from_h5py(sess, npz_file, notop = False):

# get the weights dict, and weights names
        data = np.load(npz_file)
        weights_dict = data['dict'][()]

        weights_names = list(weights_dict.keys())

# get net param
        global_variables = tf.global_variables()

        variable_names = [v.name for v in global_variables]
        variable_shapes = [v.shape for v in global_variables]

# assign the value from weights dict to net
        if notop == False:
                param_num = weights_param_num
        else:
                param_num = notop_weights_param_num
                
        for i in range(param_num):
                weights_name = weights_names[i]
                weights_value = weights_dict[weights_name]
                net_variable = global_variables[i]

                print("### load weights  ###")
                print("src weights names: ", weights_name)
                print("dst weights names: ", variable_names[i])
                print("src weights shape: ", weights_value.shape)
                print("dst weights shape: ", variable_shapes[i])

                sess.run(net_variable.assign(weights_value))

# def load_weights_param_from_npz(sess, npz_file, weights_param_lists, notop = False):

#         weights_param_list_0 = weights_param_lists[0]
#         weights_param_list_3 = weights_param_lists[3]

#         weights_0 = sess.run(weights_param_list_0[0])
#         weights_3 = sess.run(weights_param_list_3[0])

#         print("###########################", weights_0.shape)
#         print(weights_0[2][2][2])
#         print("###########################", weights_3.shape)
#         print(weights_3[2][2][2])

#         weights_param_list = weights_param_lists[0]

#         # get the weights dict, and weights names
#         data = np.load(npz_file)
#         weights_dict = data['dict'][()]
#         weights_names = list(weights_dict.keys())

#         # if notop we assign the body, it top we assign whole net
#         if notop == False:
#                 param_num = weights_param_num
#         else:
#                 param_num = notop_weights_param_num

#         for i in range(2):
#                 weights_name = weights_names[i]
#                 weights_value = weights_dict[weights_name]
#                 weights_shape = weights_value.shape

#                 net_variable = weights_param_list[i]
#                 variable_name = net_variable.name
#                 variable_shape = net_variable.shape

#                 print("### load weights  ###")
#                 print("src weights names: ", weights_name)
#                 print("dst weights names: ", variable_name)
#                 print("src weights shape: ", weights_shape)
#                 print("dst weights shape: ", variable_shape)

#                 sess.run(net_variable.assign(weights_value))

#         weights_0 = sess.run(weights_param_list_0[0])
#         weights_3 = sess.run(weights_param_list_3[0])

#         print("###########################", weights_0.shape)
#         print(weights_0[2][2][2])
#         print("###########################", weights_3.shape)
#         print(weights_3[2][2][2])

def load_weights_param_from_npz(sess, npz_file, weights_param_list, notop = False):

        # get the weights dict, and weights names
        data = np.load(npz_file)
        weights_dict = data['dict'][()]
        weights_names = list(weights_dict.keys())

        weights_test = sess.run(weights_param_list[0])
        print("###########################", weights_test.shape)
        print(weights_test[2][2][2])

        # if notop we assign the body, it top we assign whole net
        if notop == False:
                param_num = weights_param_num
        else:
                param_num = notop_weights_param_num

        for i in range(param_num):
                weights_name = weights_names[i]
                weights_value = weights_dict[weights_name]
                weights_shape = weights_value.shape

                net_variable = weights_param_list[i]
                variable_name = net_variable.name
                variable_shape = net_variable.shape

                print("### load weights  ###")
                print("src weights names: ", weights_name)
                print("dst weights names: ", variable_name)
                print("src weights shape: ", weights_shape)
                print("dst weights shape: ", variable_shape)

                sess.run(net_variable.assign(weights_value))

        weights_test = sess.run(weights_param_list[0])
        print("###########################", weights_test.shape)
        print(weights_test[2][2][2])

def check_weights_param(sess, weights_param_lists):

        weights_param_list_0 = weights_param_lists[0]
        weights_param_list_1 = weights_param_lists[1]
        weights_param_list_2 = weights_param_lists[2]        
        weights_param_list_3 = weights_param_lists[3]

        weights_0 = sess.run(weights_param_list_0[0])
        weights_1 = sess.run(weights_param_list_1[0])
        weights_2 = sess.run(weights_param_list_2[0])
        weights_3 = sess.run(weights_param_list_3[0])

        print("###########################", weights_0.shape)
        print(weights_0[2][2][2])
        print("###########################", weights_1.shape)
        print(weights_1[2][2][2])
        print("###########################", weights_2.shape)
        print(weights_2[2][2][2])
        print("###########################", weights_3.shape)
        print(weights_3[2][2][2])
