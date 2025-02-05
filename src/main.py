from load_data import *
from utils import *
from plot_helper import *
from tqdm import tqdm
from PIL import Image
import time
import sys
import os

file_dir = os.path.dirname(os.path.realpath(__file__))
# print(file_dir)

def costFunc(q):
    sum1, sum2 = 0, 0
    # print("Shape of q:", q.shape)
    # print("Shape of Exp:", Exp.shape)
    f_q = quat_mult_v(q, Exp)
    h_q = observationModel(q)
    sum1 = 0.5 * jnp.power(jnp.linalg.norm(2 * quat_log_v(quat_mult_v(quat_inv_v(q[:,1:]), f_q[:,:-1]))), 2)
    sum2 = 0.5 * jnp.linalg.norm(accel_meas[:, 1:] - h_q)**2
    SUM1_loss.append(sum1.astype(jnp.float64))
    SUM2_loss.append(sum2.astype(jnp.int32))
    return sum1 + sum2

def gradDescend(Q, costFunc, iterations=100, step_size=0.05):
    Q_iters = []
    Q_iters.append(Q[:, 1:])
    start = time.time()
    for iter in tqdm(range(iterations)):
        # print(jnp.shape(Q_iters[-1]))
        C = jax.grad(costFunc)(Q_iters[-1] + 0.001)
        Q_iters.append(Q_iters[-1]-(step_size)*C)
    print(f"Gradient descent took {time.time() - start:.2f} seconds")
    return Q_iters

def parse_args(argv):
    parsed_args = {}
    for arg in argv:
        if '=' in arg:
            key, value = arg.split('=', 1)
            if key == 'dataset':
                parsed_args['dataset'] = int(value)
            elif key == 'iterations':
                parsed_args['iterations'] = int(value)
            elif key == 'stepSize':
                parsed_args['stepSize'] = float(value)
            elif key == 'rough':
                parsed_args['rough'] = value.lower() in ['true', '1', 'yes']
    
    # Set default values if not provided
    parsed_args.setdefault('dataset', None)
    parsed_args.setdefault('iterations', 300)   # Default Iterations: 300
    parsed_args.setdefault('stepSize', 0.025)   # Default Step Size: 0.025
    parsed_args.setdefault('rough', False)
    
    return parsed_args

if __name__ == "__main__":
    ts = tic()
    SUM1_loss, SUM2_loss = [], []
    
    args = parse_args(sys.argv)

    try:
        dataset = int(args['dataset'])
    except:
        raise ValueError('Please select a dataset to process. Use: "dataset=<1-11>"')
    
    iterations = int(args['iterations'])    
    stepSize = float(args['stepSize'])
    rough = bool(args['rough'])

    ### Types of dataset: 
    #   1. imu, camera & groundtruth
    #   2. imu & groundtruth
    #   3. imu & camera   (test data)
    hasCamera = False
    hasGroundTruth = False
    dataset_folder = 'trainset'

    if dataset == 1 or dataset == 2 or dataset == 8 or dataset == 9:
        print("[Progress Report]: Dataset-Type: Panorama Stitch & GroundTruth")
        hasCamera = True
        hasGroundTruth = True

    elif dataset >= 3 and dataset < 8:
        print("[Progress Report]: Dataset-Type: GroundTruth Only")
        hasGroundTruth = True

    elif dataset > 9:
        print("[Progress Report]: Dataset-Type: Panorama Stitch Only")
        hasCamera = True
        dataset_folder = 'testset'

    print(f"[Progress Report]: Processing Dataset #{dataset} - iterations: {iterations}, step size: {stepSize}")
    
    cfile = f'{file_dir}/../data/{dataset_folder}/cam/cam{dataset}.p'
    ifile = f'{file_dir}/../data/{dataset_folder}/imu/imuRaw{dataset}.p'
    vfile = f'{file_dir}/../data/{dataset_folder}/vicon/viconRot{dataset}.p'

    ### Read imu data
    imu_data = read_data(ifile)
    dataset_count = jnp.shape(imu_data)[1]
    dataset_stamps = imu_data[0]    # time stamps of imu data

    ### Calibrate imu data
    numStaticSamples = 300  # number of static samples (No Rotation Changed) used to calibrate imu bias
    imu_biases = calculateIMUBias(numStaticSamples, imu_data)
    imu_raw = imu_data[1:]
    calibrated_accel = getAccelValue(imu_biases[:3], imu_raw[:3]) * GRAVITY # meter/sec^2
    calibrated_gyro = getGyroValue(imu_biases[3:], imu_raw[3:])             # rad/sec
    calibrated_imu_data = imu_data.copy()
    calibrated_imu_data[1:4] = calibrated_accel
    calibrated_imu_data[4:] = calibrated_gyro
    # print(calibrated_imu_data)

    ### Read Camera/GroundTruth data depending on the dataset
    cam_data = None
    vicon_data = None
    if hasCamera:
        cam_data = read_data(cfile)
        # print(cam_data.keys())
    if hasGroundTruth:
        vicon_data = read_data(vfile)
        # print(vicon_data.keys())
    print("[Progress Report]: Data Calibrated")

    ### Measurements Extract
    # Add gravity on accel measurements
    accel_meas = (
        jnp.multiply(jnp.array(calibrated_imu_data[1:4, :]), jnp.array([-1, -1, -1])[:, jnp.newaxis]) + jnp.array([0.,0.,-GRAVITY])[:, jnp.newaxis]
    )                                           # (3, n)
    wt = calibrated_imu_data[4:, :]             # (3, n)
    
    ### Generate Prediction Trajectory form Initial Guess of Quaternion:(1,0,0,0)
    Exp = []
    quats_prior = [jnp.array([1,0,0,0], dtype='float64')]
    accel_prior = []

    ### Motion Model Prediction
    for i in range(dataset_count - 1):
        qt = quat_vectorize(quats_prior[i])
        taut = imu_data[0, i+1] - imu_data[0, i]
        exp, qt_1_prior = motionModel(qt, taut, wt[:, i])
        Exp.append(quat_toRow(exp))
        quats_prior.append(quat_toRow(qt_1_prior))
    Exp = jnp.transpose(jnp.array(Exp))                 # (4, n-1)
    quats_prior = jnp.transpose(jnp.array(quats_prior)) # (4, n)

    ### Observation Model Prediction
    accel_prior = observationModel(quats_prior)    # (3, n)

    ### Run Gradient Descent
    print('[Progress Report]: Started Training with Gradient Descent')
    Q_iters = jnp.array(gradDescend(quats_prior, costFunc, iterations, stepSize))
    print(f'Loss started at: {SUM2_loss[0].astype(jnp.int32)}')
    print(f'Loss ended at: {SUM2_loss[-1].astype(jnp.int32)}')

    ### Plotting the graph
    print('[Progress Report]: Plotting and Saving Graphs')
    R = plot_euler_angles(quats_prior, vicon_data, Q_iters[-1], dataset)
    plot_acceleration(accel_prior, accel_meas, observationModel(Q_iters[-1]), dataset)

    if not hasGroundTruth:
        vicon_data = {'ts': imu_data[:, :R.shape[1]], 'rots': np.zeros((3, 3, R.shape[1]))}
        for i, r in enumerate(R.T):
            vicon_data['rots'][:, :, i] = transforms3d.euler.euler2mat(r[0], r[1], r[2])

    if hasCamera:
        print('[Progress Report]: Starting construction of panorama')
        image = create_panorama(vicon_data, cam_data, rough)
        Image.fromarray(image.astype(jnp.uint8)).save(f'panorama_rough_{dataset}.jpg' if rough else f'panorama_{dataset}.jpg')
        print('[Progress Report]: Image stored')

    toc(ts, "Program process")