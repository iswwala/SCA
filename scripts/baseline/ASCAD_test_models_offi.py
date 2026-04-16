import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
	print('no display found. Using non-interactive Agg backend')
	mpl.use('Agg')
else:
	mpl.use('TkAgg')
import os.path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import ast
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model

# The AES SBox that we will use to compute the rank
AES_Sbox = np.array([
		0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
		0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
		0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
		0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
		0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
		0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
		0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
		0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
		0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
		0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
		0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
		0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
		0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
		0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
		0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
		0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
		])

# Two Tables to process a field multplication over GF(256): a*b = alog (log(a) + log(b) mod 255)
log_table=[ 0, 0, 25, 1, 50, 2, 26, 198, 75, 199, 27, 104, 51, 238, 223, 3,
	100, 4, 224, 14, 52, 141, 129, 239, 76, 113, 8, 200, 248, 105, 28, 193,
	125, 194, 29, 181, 249, 185, 39, 106, 77, 228, 166, 114, 154, 201, 9, 120,
	101, 47, 138, 5, 33, 15, 225, 36, 18, 240, 130, 69, 53, 147, 218, 142,
	150, 143, 219, 189, 54, 208, 206, 148, 19, 92, 210, 241, 64, 70, 131, 56,
	102, 221, 253, 48, 191, 6, 139, 98, 179, 37, 226, 152, 34, 136, 145, 16,
	126, 110, 72, 195, 163, 182, 30, 66, 58, 107, 40, 84, 250, 133, 61, 186,
	43, 121, 10, 21, 155, 159, 94, 202, 78, 212, 172, 229, 243, 115, 167, 87,
	175, 88, 168, 80, 244, 234, 214, 116, 79, 174, 233, 213, 231, 230, 173, 232,
	44, 215, 117, 122, 235, 22, 11, 245, 89, 203, 95, 176, 156, 169, 81, 160,
	127, 12, 246, 111, 23, 196, 73, 236, 216, 67, 31, 45, 164, 118, 123, 183,
	204, 187, 62, 90, 251, 96, 177, 134, 59, 82, 161, 108, 170, 85, 41, 157,
	151, 178, 135, 144, 97, 190, 220, 252, 188, 149, 207, 205, 55, 63, 91, 209,
	83, 57, 132, 60, 65, 162, 109, 71, 20, 42, 158, 93, 86, 242, 211, 171,
	68, 17, 146, 217, 35, 32, 46, 137, 180, 124, 184, 38, 119, 153, 227, 165,
	103, 74, 237, 222, 197, 49, 254, 24, 13, 99, 140, 128, 192, 247, 112, 7 ]

alog_table =[1, 3, 5, 15, 17, 51, 85, 255, 26, 46, 114, 150, 161, 248, 19, 53,
	95, 225, 56, 72, 216, 115, 149, 164, 247, 2, 6, 10, 30, 34, 102, 170,
	229, 52, 92, 228, 55, 89, 235, 38, 106, 190, 217, 112, 144, 171, 230, 49,
	83, 245, 4, 12, 20, 60, 68, 204, 79, 209, 104, 184, 211, 110, 178, 205,
	76, 212, 103, 169, 224, 59, 77, 215, 98, 166, 241, 8, 24, 40, 120, 136,
	131, 158, 185, 208, 107, 189, 220, 127, 129, 152, 179, 206, 73, 219, 118, 154,
	181, 196, 87, 249, 16, 48, 80, 240, 11, 29, 39, 105, 187, 214, 97, 163,
	254, 25, 43, 125, 135, 146, 173, 236, 47, 113, 147, 174, 233, 32, 96, 160,
	251, 22, 58, 78, 210, 109, 183, 194, 93, 231, 50, 86, 250, 21, 63, 65,
	195, 94, 226, 61, 71, 201, 64, 192, 91, 237, 44, 116, 156, 191, 218, 117,
	159, 186, 213, 100, 172, 239, 42, 126, 130, 157, 188, 223, 122, 142, 137, 128,
	155, 182, 193, 88, 232, 35, 101, 175, 234, 37, 111, 177, 200, 67, 197, 84,
	252, 31, 33, 99, 165, 244, 7, 9, 27, 45, 119, 153, 176, 203, 70, 202,
	69, 207, 74, 222, 121, 139, 134, 145, 168, 227, 62, 66, 198, 81, 243, 14,
	18, 54, 90, 238, 41, 123, 141, 140, 143, 138, 133, 148, 167, 242, 13, 23,
	57, 75, 221, 124, 132, 151, 162, 253, 28, 36, 108, 180, 199, 82, 246, 1 ]

# Multiplication function in GF(2^8)
def multGF256(a,b):
	if (a==0) or (b==0):
		return 0
	else:
		return alog_table[(log_table[a]+log_table[b]) %255]

def check_file_exists(file_path):
	file_path = os.path.normpath(file_path)
	if os.path.exists(file_path) == False:
		print("Error: provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)
	return

def load_sca_model(model_file):
	check_file_exists(model_file)
	try:
		model = load_model(model_file)
	except:
		print("Error: can't load Keras model file '%s'" % model_file)
		sys.exit(-1)
	return model

# Compute the rank of the real key for a give set of predictions
def rank(predictions, metadata, real_key, min_trace_idx, max_trace_idx, last_key_bytes_proba, target_byte, simulated_key):
	# Compute the rank
	if len(last_key_bytes_proba) == 0:
		# If this is the first rank we compute, initialize all the estimates to zero
		key_bytes_proba = np.zeros(256)
	else:
		# This is not the first rank we compute: we optimize things by using the
		# previous computations to save time!
		key_bytes_proba = last_key_bytes_proba

	for p in range(0, max_trace_idx-min_trace_idx):
		# Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.
		plaintext = metadata[min_trace_idx + p]['plaintext'][target_byte]
		key = metadata[min_trace_idx + p]['key'][target_byte]
		for i in range(0, 256):
			# Our candidate key byte probability is the sum of the predictions logs
			if (simulated_key!=1):
				proba = predictions[p][AES_Sbox[plaintext ^ i]]
			else:
				proba = predictions[p][AES_Sbox[plaintext ^ key ^ i]]
			if proba != 0:
				key_bytes_proba[i] += np.log(proba)
			else:
				# We do not want an -inf here, put a very small epsilon
				# that correspondis to a power of our min non zero proba
				min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
				if len(min_proba_predictions) == 0:
					print("Error: got a prediction with only zeroes ... this should not happen!")
					sys.exit(-1)
				min_proba = min(min_proba_predictions)
				key_bytes_proba[i] += np.log(min_proba**2)
	# Now we find where our real key candidate lies in the estimation.
	# We do this by sorting our estimates and find the rank in the sorted array.
	sorted_proba = np.array(list(map(lambda a : key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
	real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
	return (real_key_rank, key_bytes_proba)

def timestamped_filename(model_file, ascad_database, save_file):
	stamp = datetime.now().strftime('%m%d%H%M')
	model_name = os.path.splitext(os.path.basename(model_file))[0]
	db_name = os.path.splitext(os.path.basename(ascad_database))[0]
	dirname = os.path.dirname(save_file)
	ext = os.path.splitext(save_file)[1] if os.path.splitext(save_file)[1] else '.png'
	filename = f"{model_name}_{db_name}_{stamp}{ext}"
	return os.path.join(dirname, filename) if dirname else filename

def full_ranks(predictions, dataset, metadata, min_trace_idx, max_trace_idx, rank_step, target_byte, simulated_key):
	print("Computing rank for targeted byte {}".format(target_byte))
	# Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.
	if (simulated_key!=1):
		real_key = metadata[0]['key'][target_byte]
	else:
		real_key = 0
	# Check for overflow
	if max_trace_idx > dataset.shape[0]:
		print("Error: asked trace index %d overflows the total traces number %d" % (max_trace_idx, dataset.shape[0]))
		sys.exit(-1)
	index = np.arange(min_trace_idx+rank_step, max_trace_idx, rank_step)
	f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
	key_bytes_proba = []
	for t, i in zip(index, range(0, len(index))):
		real_key_rank, key_bytes_proba = rank(predictions[t-rank_step:t], metadata, real_key, t-rank_step, t, key_bytes_proba, target_byte, simulated_key)
		f_ranks[i] = [t - min_trace_idx, real_key_rank]
	return f_ranks

#### ASCAD helper to load profiling and attack data (traces and labels)
# Loads the profiling and attack datasets from the ASCAD
# database
def load_ascad(ascad_database_file, load_metadata=False):
	check_file_exists(ascad_database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file	 = h5py.File(ascad_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
		sys.exit(-1)
	# Load profiling traces
	X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
	# Load profiling labels
	Y_profiling = np.array(in_file['Profiling_traces/labels'])
	# Load attacking traces
	X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
	# Load attacking labels
	Y_attack = np.array(in_file['Attack_traces/labels'])
	if load_metadata == False:
		return (X_profiling, Y_profiling), (X_attack, Y_attack)
	else:
		return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])

# Compute Pr(Sbox(p^k)*alpha|t)
def proba_dissect_beta(proba_sboxmuladd, proba_beta):
	proba = np.zeros(proba_sboxmuladd.shape)
	for j in range(proba_beta.shape[1]):
		proba_sboxdeadd = proba_sboxmuladd[:, [(beta^j) for beta in range(256)]]
		proba[:,j] = np.sum(proba_sboxdeadd*proba_beta, axis=1)
	return proba

# Compute Pr(Sbox(p^k)|t)
def proba_dissect_alpha(proba_sboxmul, proba_alpha):
	proba = np.zeros(proba_sboxmul.shape)
	for j in range(proba_alpha.shape[1]):
		proba_sboxdemul = proba_sboxmul[:, [multGF256(alpha,j) for alpha in range(256)]]
		proba[:,j] = np.sum(proba_sboxdemul*proba_alpha, axis=1)
	return proba

# Compute Pr(Sbox(p[permind]^k[permind])|t)
def proba_dissect_permind(proba_x, proba_permind, j):
	proba = np.zeros((proba_x.shape[0], proba_x.shape[2]))
	for s in range(proba_x.shape[2]):
		proba_1 = proba_x[:,:,s]
		proba_2 = proba_permind[:,:,j]
		proba[:,s] = np.sum(proba_1*proba_2, axis=1)
	return proba

# Compute Pr(Sbox(p^k)|t) by a recombination of the guessed probilities, with the permIndices known during the profiling phase
def multilabel_predict(predictions):
	predictions_alpha = predictions[0]
	predictions_beta = predictions[1]
	predictions_unshuffledsboxmuladd = []
	predictions_permind = []
	for i in range(16):
		predictions_unshuffledsboxmuladd.append(predictions[2+i])
		predictions_permind.append(predictions[2+16+i])

	predictions_unshuffledsboxmul = []
	print("Computing multiplicative masked sbox probas with shuffle...")
	for i in range(16):
		predictions_unshuffledsboxmul.append(proba_dissect_beta(predictions_unshuffledsboxmuladd[i], predictions_beta))

	print("Computing sbox probas with shuffle...")
	predictions_unshuffledsbox = []
	for i in range(16):
		predictions_unshuffledsbox.append(proba_dissect_alpha(predictions_unshuffledsboxmul[i], predictions_alpha))

	predictions_unshuffledsbox_v = np.array(predictions_unshuffledsbox)
	predictions_permind_v = np.array(predictions_permind)
	predictions_unshuffledsbox_v = np.moveaxis(predictions_unshuffledsbox_v, [0,1,2], [1,0,2])
	predictions_permind_v = np.moveaxis(predictions_permind_v, [0,1,2], [1,0,2])
	predictions_sbox = []
	print("Computing sbox probas...")
	for i in range(16):
		predictions_sbox.append(proba_dissect_permind(predictions_unshuffledsbox_v, predictions_permind_v, i))

	return predictions_sbox

# Compute Pr(Sbox(p^k)|t) by a recombination of the guessed probilities without taking the shuffling into account
def multilabel_without_permind_predict(predictions):
	predictions_alpha = predictions[0]
	predictions_beta = predictions[1]
	predictions_sboxmuladd = []
	for i in range(16):
		predictions_sboxmuladd.append(predictions[2+i])

	predictions_sboxmul = []
	print("Computing multiplicative masked sbox...")
	for i in range(16):
		predictions_sboxmul.append(proba_dissect_beta(predictions_sboxmuladd[i], predictions_beta))

	print("Computing sbox probas...")
	predictions_sbox = []
	for i in range(16):
		predictions_sbox.append(proba_dissect_alpha(predictions_sboxmul[i], predictions_alpha))

	return predictions_sbox

def read_parameters_from_file(param_filename):
	#read parameters for the extract_traces function from given filename
	#TODO: sanity checks on parameters
	param_file = open(param_filename,"r")

	#FIXME: replace eval() by ast.linear_eval()
	my_parameters= eval(param_file.read())

	model_file = my_parameters["model_file"]
	ascad_database = my_parameters["ascad_database"]
	num_traces = my_parameters["num_traces"]
	target_byte = 2
	if ("target_byte" in my_parameters):
		target_byte = my_parameters["target_byte"]
	multilabel = 0
	if ("multilabel" in my_parameters):
		multilabel = my_parameters["multilabel"]
	simulated_key = 0
	if ("simulated_key" in my_parameters):
		simulated_key = my_parameters["simulated_key"]
	save_file = ""
	if ("save_file" in my_parameters):
		save_file = my_parameters["save_file"]

	return model_file, ascad_database, num_traces, target_byte, multilabel, simulated_key, save_file


def rank_with_metrics(predictions, metadata, real_key, min_trace_idx, max_trace_idx, 
                       last_key_bytes_proba, target_byte, simulated_key):
    """计算排名，同时返回中间指标"""
    if len(last_key_bytes_proba) == 0:
        key_bytes_proba = np.zeros(256)
    else:
        key_bytes_proba = last_key_bytes_proba.copy()
    
    # 记录每条痕迹的信息
    trace_metrics = []
    
    for p in range(0, max_trace_idx - min_trace_idx):
        plaintext = metadata[min_trace_idx + p]['plaintext'][target_byte]
        key = metadata[min_trace_idx + p]['key'][target_byte]
        
        # 记录这条痕迹的信息
        trace_info = {
            'trace_idx': min_trace_idx + p,
            'plaintext': int(plaintext),
            'true_key': int(key),
            'true_label': int(AES_Sbox[plaintext ^ key]),
            'probs': predictions[p].copy()
        }
        
        for i in range(0, 256):
            if simulated_key != 1:
                proba = predictions[p][AES_Sbox[plaintext ^ i]]
            else:
                proba = predictions[p][AES_Sbox[plaintext ^ key ^ i]]
            if proba != 0:
                key_bytes_proba[i] += np.log(proba)
            else:
                min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                if len(min_proba_predictions) == 0:
                    print("Error: got a prediction with only zeroes ... this should not happen!")
                    sys.exit(-1)
                min_proba = min(min_proba_predictions)
                key_bytes_proba[i] += np.log(min_proba**2)
        
        # 记录当前累积得分
        trace_info['cumulative_scores'] = key_bytes_proba.copy()
        trace_metrics.append(trace_info)
    
    sorted_proba = np.array(list(map(lambda a: key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    
    return real_key_rank, key_bytes_proba, trace_metrics


def full_ranks_with_metrics(predictions, dataset, metadata, min_trace_idx, max_trace_idx, 
                             rank_step, target_byte, simulated_key):
    """计算排名，同时收集详细的指标"""
    print("Computing rank for targeted byte {}".format(target_byte))
    
    if simulated_key != 1:
        real_key = metadata[0]['key'][target_byte]
    else:
        real_key = 0
    
    if max_trace_idx > dataset.shape[0]:
        print("Error: asked trace index %d overflows the total traces number %d" % (max_trace_idx, dataset.shape[0]))
        sys.exit(-1)
    
    index = np.arange(min_trace_idx + rank_step, max_trace_idx, rank_step)
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)
    key_bytes_proba = []
    all_trace_metrics = []
    
    for t, i in zip(index, range(0, len(index))):
        real_key_rank, key_bytes_proba, trace_metrics = rank_with_metrics(
            predictions[t-rank_step:t], metadata, real_key, 
            t-rank_step, t, key_bytes_proba, target_byte, simulated_key
        )
        f_ranks[i] = [t - min_trace_idx, real_key_rank]
        all_trace_metrics.extend(trace_metrics)
    
    return f_ranks, all_trace_metrics, key_bytes_proba


def compute_detailed_metrics(predictions, metadata, num_traces, target_byte):
    """计算详细的评估指标"""
    metrics = {}
    
    # 确保 predictions 是 numpy 数组
    predictions = np.array(predictions)
    
    # 1. 每条痕迹的正确密钥概率
    correct_probs = []
    for i in range(num_traces):
        plaintext = metadata[i]['plaintext'][target_byte]
        key = metadata[i]['key'][target_byte]
        true_label = AES_Sbox[plaintext ^ key]
        correct_probs.append(predictions[i][true_label])
    
    # 转换为 numpy 数组
    correct_probs = np.array(correct_probs)
    metrics['correct_probs'] = correct_probs
    metrics['correct_probs_mean'] = np.mean(correct_probs)
    metrics['correct_probs_std'] = np.std(correct_probs)
    metrics['correct_probs_min'] = np.min(correct_probs)
    metrics['correct_probs_max'] = np.max(correct_probs)
    
    # 2. 每条痕迹的最大概率
    max_probs = np.max(predictions, axis=1)
    metrics['max_probs'] = max_probs
    metrics['max_probs_mean'] = np.mean(max_probs)
    metrics['max_probs_std'] = np.std(max_probs)
    
    # 3. 正确概率的排名（每条痕迹单独）
    correct_ranks = []
    for i in range(num_traces):
        plaintext = metadata[i]['plaintext'][target_byte]
        key = metadata[i]['key'][target_byte]
        true_label = AES_Sbox[plaintext ^ key]
        rank = np.sum(predictions[i] > predictions[i][true_label])
        correct_ranks.append(rank)
    
    correct_ranks = np.array(correct_ranks)
    metrics['correct_ranks'] = correct_ranks
    metrics['correct_ranks_mean'] = np.mean(correct_ranks)
    metrics['correct_ranks_std'] = np.std(correct_ranks)
    
    # 4. 正确概率 vs 最大概率的比值
    metrics['confidence_ratio'] = correct_probs / (max_probs + 1e-8)
    metrics['confidence_ratio_mean'] = np.mean(metrics['confidence_ratio'])
    
    # 5. 找出低置信度的痕迹
    low_confidence_idx = np.where(correct_probs < 0.01)[0]
    metrics['low_confidence_count'] = len(low_confidence_idx)
    metrics['low_confidence_ratio'] = len(low_confidence_idx) / num_traces
    
    # 6. 找出预测错误的痕迹（单条痕迹排名 > 0）
    wrong_predictions = np.where(correct_ranks > 0)[0]
    metrics['wrong_predictions_count'] = len(wrong_predictions)
    metrics['wrong_predictions_ratio'] = len(wrong_predictions) / num_traces
    
    return metrics

def plot_detailed_metrics(metrics, save_file_prefix):
    """绘制详细的指标图表"""
    
    # 图1：正确概率分布
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(metrics['correct_probs'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Correct key probability')
    plt.ylabel('Frequency')
    plt.title(f'Correct Key Probability Distribution\n(mean={metrics["correct_probs_mean"]:.4f})')
    plt.axvline(metrics['correct_probs_mean'], color='r', linestyle='--', label='mean')
    plt.legend()
    
    # 图2：正确概率随痕迹索引变化
    plt.subplot(2, 3, 2)
    plt.plot(metrics['correct_probs'][:5000], alpha=0.5)
    plt.xlabel('Trace index')
    plt.ylabel('Correct key probability')
    plt.title('Correct Key Probability over Time')
    plt.axhline(1/256, color='r', linestyle='--', label='random (1/256)')
    plt.legend()
    
    # 图3：最大概率分布
    plt.subplot(2, 3, 3)
    plt.hist(metrics['max_probs'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Max probability')
    plt.ylabel('Frequency')
    plt.title(f'Max Probability Distribution\n(mean={metrics["max_probs_mean"]:.4f})')
    plt.axvline(metrics['max_probs_mean'], color='r', linestyle='--', label='mean')
    plt.legend()
    
    # 图4：正确概率的排名分布
    plt.subplot(2, 3, 4)
    plt.hist(metrics['correct_ranks'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Rank of correct key (single trace)')
    plt.ylabel('Frequency')
    plt.title(f'Correct Key Rank Distribution\n(mean={metrics["correct_ranks_mean"]:.1f})')
    plt.axvline(metrics['correct_ranks_mean'], color='r', linestyle='--', label='mean')
    plt.legend()
    
    # 图5：正确概率 vs 最大概率比值
    plt.subplot(2, 3, 5)
    plt.hist(metrics['confidence_ratio'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Confidence ratio (correct / max)')
    plt.ylabel('Frequency')
    plt.title(f'Confidence Ratio Distribution\n(mean={metrics["confidence_ratio_mean"]:.4f})')
    plt.axvline(metrics['confidence_ratio_mean'], color='r', linestyle='--', label='mean')
    plt.legend()
    
    # 图6：低置信度痕迹比例
    labels = ['High Confidence\n(prob>=0.01)', 'Low Confidence\n(prob<0.01)']
    sizes = [100 - metrics['low_confidence_ratio'] * 100, metrics['low_confidence_ratio'] * 100]
    plt.subplot(2, 3, 6)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'Low Confidence Traces\n(count={metrics["low_confidence_count"]})')
    
    plt.tight_layout()
    plt.savefig(f'{save_file_prefix}_detailed_metrics.png', dpi=150)
    plt.close()
    print(f"Saved detailed metrics plot to {save_file_prefix}_detailed_metrics.png")


def check_model_detailed(model_file, ascad_database, num_traces=2000, target_byte=2, 
                          multilabel=0, simulated_key=0, save_file=""):
    """带详细指标的模型评估"""
    check_file_exists(model_file)
    check_file_exists(ascad_database)
    
    # 加载数据
    (X_profiling, Y_profiling), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) = load_ascad(
        ascad_database, load_metadata=True)
    
    # 加载模型
    model = load_sca_model(model_file)
    
    # 获取输入层形状
    input_layer_shape = model.get_layer(index=0).input_shape[0]
    if isinstance(model.get_layer(index=0).input_shape, list):
        input_layer_shape = model.get_layer(index=0).input_shape[0]
    else:
        input_layer_shape = model.get_layer(index=0).input_shape
    
    # 形状检查
    if input_layer_shape[1] != len(X_attack[0, :]):
        print("Error: model input shape %d instead of %d is not expected ..." % 
              (input_layer_shape[1], len(X_attack[0, :])))
        sys.exit(-1)
    
    # 调整输入形状
    if len(input_layer_shape) == 2:
        input_data = X_attack[:num_traces, :]
    elif len(input_layer_shape) == 3:
        input_data = X_attack[:num_traces, :]
        input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)
    
    # 预测
    print("Predicting...")
    predictions = model.predict(input_data)
    
    # 计算详细指标
    print("Computing detailed metrics...")
    metrics = compute_detailed_metrics(predictions, Metadata_attack[:num_traces], num_traces, target_byte)
    
    # 打印详细指标
    print("\n" + "="*60)
    print("DETAILED METRICS")
    print("="*60)
    print(f"Correct Key Probability:")
    print(f"  Mean: {metrics['correct_probs_mean']:.6f}")
    print(f"  Std: {metrics['correct_probs_std']:.6f}")
    print(f"  Min: {metrics['correct_probs_min']:.6f}")
    print(f"  Max: {metrics['correct_probs_max']:.6f}")
    print(f"  Random baseline: {1/256:.6f}")
    print()
    print(f"Max Probability:")
    print(f"  Mean: {metrics['max_probs_mean']:.6f}")
    print(f"  Std: {metrics['max_probs_std']:.6f}")
    print()
    print(f"Correct Key Rank (per trace):")
    print(f"  Mean: {metrics['correct_ranks_mean']:.1f}")
    print(f"  Std: {metrics['correct_ranks_std']:.1f}")
    print(f"  Random baseline: 127.5")
    print()
    print(f"Confidence Ratio (correct / max):")
    print(f"  Mean: {metrics['confidence_ratio_mean']:.4f}")
    print()
    print(f"Low Confidence Traces (P_correct < 0.01):")
    print(f"  Count: {metrics['low_confidence_count']} / {num_traces}")
    print(f"  Ratio: {metrics['low_confidence_ratio']*100:.2f}%")
    print()
    print(f"Wrong Predictions (rank > 0):")
    print(f"  Count: {metrics['wrong_predictions_count']} / {num_traces}")
    print(f"  Ratio: {metrics['wrong_predictions_ratio']*100:.2f}%")
    
    # 保存详细指标图
    save_file_base = os.path.splitext(save_file)[0] if save_file else "metrics"
    plot_detailed_metrics(metrics, save_file_base)
    
    # 计算GE排名
    print("\nComputing GE ranks...")
    if multilabel != 0:
        # 多标签处理...
        pass
    else:
        ranks, all_trace_metrics, final_scores = full_ranks_with_metrics(
            predictions, X_attack, Metadata_attack, 0, num_traces, 10, target_byte, simulated_key)
        
        # 绘制GE曲线
        x = [ranks[i][0] for i in range(ranks.shape[0])]
        y = [ranks[i][1] for i in range(ranks.shape[0])]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Number of traces')
        plt.ylabel('Guessing Entropy (Rank)')
        plt.title(f'GE Curve\nModel: {os.path.basename(model_file)}\nDatabase: {os.path.basename(ascad_database)}')
        plt.grid(True, alpha=0.3)
        
        # 添加最终GE标注
        final_ge = y[-1] if y else 255
        plt.annotate(f'Final GE: {final_ge}', xy=(x[-1], y[-1]), xytext=(x[-1]*0.7, y[-1]*0.7),
                     arrowprops=dict(arrowstyle='->', color='red'))
        
        if save_file:
            save_file_ts = timestamped_filename(model_file, ascad_database, save_file)
            plt.savefig(save_file_ts, dpi=150)
            print(f"Saved GE curve to {save_file_ts}")
        else:
            plt.show()
        
        # 打印最终GE
        print(f"\nFinal GE after {num_traces} traces: {final_ge}")
    
    predicted_labels = np.argmax(predictions, axis=1)
    unique, counts = np.unique(predicted_labels, return_counts=True)
    print("Predicted label distribution:")
    for u, c in zip(unique[:20], counts[:20]):
        print(f"  Label {u}: {c} times")

    # 检查是否有单一标签占主导
    print(f"Most common label: {unique[np.argmax(counts)]} with {np.max(counts)} times")


# 主程序入口
if __name__ == "__main__":
    if len(sys.argv) != 2:
        model_file = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_trained_models/cnn_best_ascad_desync0_epochs75_classes256_batchsize200.h5"
        ascad_database = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
        num_traces = 2000
        target_byte = 2
        multilabel = 0
        simulated_key = 0
        save_file = ""
    else:
        model_file, ascad_database, num_traces, target_byte, multilabel, simulated_key, save_file = read_parameters_from_file(sys.argv[1])
    
    check_model_detailed(model_file, ascad_database, num_traces, target_byte, multilabel, simulated_key, save_file)
    

    try:
        input("Press enter to exit ...")
    except SyntaxError:
        pass