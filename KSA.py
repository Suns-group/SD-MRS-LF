import math
import random
import numpy as np
import copy


def neighbours(grid_w, grid_h, num):
    lign = get_lign_number(num, grid_w)
    n = []
    if num + (grid_w - 1) != lign * grid_w:
        n.append(num - 1)

    if num % grid_w != 0:
        n.append(num + 1)

    if num - 1 not in range(grid_w):
        n.append(num - grid_w)

    if num - 1 not in range(num, grid_h * grid_w):
        n.append(num + grid_w)

    if num - 1 not in range(grid_w) and num + (grid_w - 1) != lign * grid_w:
        n.append(num - (grid_w + 1))

    if num % grid_w != 0 and num - 1 not in range(num, grid_h * grid_w):
        n.append(num + (grid_w + 1))

    if num % grid_w != 0 and num - 1 not in range(grid_w):
        n.append(num - (grid_w - 1))

    if num + (grid_w - 1) != lign * grid_w and num - 1 not in range(num,
                                                                    grid_h * grid_w):
        n.append(num + (grid_w - 1))

    to_delete = []
    for i in range(len(n)):
        if n[i] > grid_w * grid_h:
            to_delete.append(n[i])
    for elt in to_delete:
        n.remove(elt)

    neighbours = n

    return neighbours


def generate_grid(grid_w, grid_h):
    grid = []
    for i in range(grid_w * grid_h):
        grid.append(i + 1)
    return grid


def generate_solution(grid_w, grid_h, landing_point, takeoff_point, grid):
    solution = []

    for point in grid:
        if point != landing_point and point != takeoff_point:
            solution.append(point)

    random.shuffle(solution)
    solution.insert(0, landing_point)
    solution.insert(len(grid) - 1, takeoff_point)

    solution = arrange_solution(solution, grid_w, grid_h, landing_point, takeoff_point)
    return solution


def arrange_solution(solution, grid_w, grid_h, landing_point, takeoff_point):
    for i in range(len(solution) - 1):
        current_point = solution[i]
        next_point = solution[i + 1]
        current_point_neighbours = neighbours(grid_w, grid_h, current_point)
        if landing_point in current_point_neighbours:
            current_point_neighbours.remove(landing_point)
        if takeoff_point in current_point_neighbours:
            current_point_neighbours.remove(takeoff_point)

        neighbours_to_remove = []
        for j in range(len(current_point_neighbours)):
            if current_point_neighbours[j] in solution[
                                              0:solution.index(current_point)]:
                neighbours_to_remove.append(current_point_neighbours[j])
        for neighbours_ in neighbours_to_remove:
            current_point_neighbours.remove(neighbours_)

        if next_point not in current_point_neighbours and len(
                current_point_neighbours) > 0:
            choosed_neighbour = random.choice(current_point_neighbours)
            idx = solution.index(choosed_neighbour)
            solution[idx] = next_point
            solution[i + 1] = choosed_neighbour
    return solution


def get_lign_number(num, grid_w):
    if (num / grid_w % 1) > 0.0:
        return math.floor(num / grid_w) + 1
    else:
        return math.floor(num / grid_w)


def get_col_number(num, grid_w):
    mod = num % grid_w
    if mod == 0:
        return grid_w
    else:
        return mod


def get_coords_x_number(num, grid_w):
    col_num = get_col_number(num, grid_w) - 1
    if col_num < 0:
        return 0
    else:
        return col_num


def get_coords_y_number(num, grid_w):
    lign_num = get_lign_number(num, grid_w) - 1
    if lign_num < 0:
        return 0
    else:
        return lign_num


def mAh_to_Ah(val):
    """Convert mAH to Ah"""
    return val * 0.001


def get_vector_from(A=0, B=0, Distance_Unit=0, grid_w=0):
    """Create vector from 2 point A & B"""
    P1 = [0, 0]
    P1[0] = get_col_number(A, grid_w) - 1
    P1[1] = get_lign_number(A, grid_w) - 1
    if P1[0] < 0:
        P1[0] = 0
    if P1[1] < 0:
        P1[1] = 0
    P1[0] = P1[0] * Distance_Unit
    P1[1] = P1[1] * Distance_Unit

    P2 = [0, 0]
    P2[0] = get_col_number(B, grid_w) - 1
    P2[1] = get_lign_number(B, grid_w) - 1
    if P2[0] < 0:
        P2[0] = 0
    if P2[1] < 0:
        P2[1] = 0
    P2[0] = P2[0] * Distance_Unit
    P2[1] = P2[1] * Distance_Unit

    return [P2[0] - P1[0], P2[1] - P1[1]]


def get_vector_magnitude(V):
    return math.sqrt((V[0]) ** 2 + (V[1]) ** 2)


def normalize(V_xy):
    m = get_vector_magnitude(V_xy)
    #print(m, V_xy)
    X = float(V_xy[0]) / m
    Y = float(V_xy[1]) / m
    return [X, Y]


def get_speed_vector(speed=0, V_xy=[0, 0]):
    X = 0
    Y = 0
    m = get_vector_magnitude(V_xy)
    if m > 0:
        X = (speed * V_xy[0]) / m
        Y = (speed * V_xy[1]) / m

    return [X, Y]


def get_distance_between(A, B, Distance_unit, grid_w):
    P = get_vector_from(A, B, Distance_unit, grid_w)
    return math.sqrt(P[0] ** 2 + P[1] ** 2)


def f(z, omega_):
    return 2 * z ** 5 + 3 * omega_ * z ** 4 - 2 * z - omega_


def df(z, omega_):
    return 10 * z ** 4 + 12 * omega_ * z ** 3 - 2


def newtons_method(f, df, omega_, x0, e):
    while abs(f(x0, omega_)) > e:
        x0 = x0 - f(x0, omega_) / df(x0, omega_)
    return x0


def generate_grid(grid_w, grid_h):
    grid = []
    for i in range(grid_w * grid_h):
        grid.append(i + 1)
    grid.append(1)
    return grid


def costheta(u, v):
    u_n = normalize(u)
    v_n = normalize(v)
    dotP = u_n[0] * v_n[0] + u_n[1] * v_n[1]
    val = dotP / (get_vector_magnitude(u_n) * get_vector_magnitude(v_n))
    val = min(val, 1)  # keep the value between -1 and 1
    val = max(val, -1)
    theta = math.acos(val)
    ct = math.cos(theta)
    return ct


def theta(u, v):
    u_n = normalize(u)
    v_n = normalize(v)
    dotP = u_n[0] * v_n[0] + u_n[1] * v_n[1]
    val = dotP / (get_vector_magnitude(u_n) * get_vector_magnitude(v_n))
    val = min(val, 1)  # keep the value between -1 and 1
    val = max(val, -1)
    angle = math.acos(val)
    return angle


def comp_vopt(wind_speed, v0, vec, wind_dir):
    ct = costheta(vec, wind_dir)
    omega = wind_speed * ct / v0
    z = newtons_method(f, df, omega, 20, .001)
    v_opt = v0 * z
    return v_opt


def comp_energy_xy(x, y, v0, a, b, wind_dir, wind_speed, dist_unit, grid_w):
    vec = get_vector_from(x, y, dist_unit, grid_w)
    vopt = comp_vopt(wind_speed, v0, vec, wind_dir)
    ct = costheta(vec, wind_dir)
    dist = get_distance_between(x, y, dist_unit, grid_w)
    e = (a * vopt ** 3 + b / vopt) * dist / (vopt + wind_speed * ct)
    time = dist / (vopt + wind_speed * ct)
    return e, time


def comp_energy_sol(solution, v0, a, b, wind_dir, wind_speed, dist_unit, grid_w):
    energy = 0
    time = 0
    n = len(solution)
    for i in range(n - 1):
        e = comp_energy_xy(solution[i], solution[i + 1], v0, a, b, wind_dir, wind_speed, dist_unit, grid_w)
        energy += e[0]
        time += e[1]
    return energy, time


### Fonction objective
def f_objective(solution, charge_points, takeoff_point,
                maximum_energy_of_the_drone, distance_unit, grid_w, wind_speed,
                v0, wind_vector, alpha, beta):
    sol = copy.copy(solution[0])

    pc_solution = []
    pc_used = []
    energies = []
    dr_dead = []
    for elt in sol:
        if elt in charge_points:
            pc_solution.append(1)
        else:
            pc_solution.append(0)
        pc_used.append(0)
        energies.append(0)
        dr_dead.append(0)
    solution_representation = [
        sol,
        pc_solution,
        pc_used,
        energies,
        dr_dead,
    ]
    tmp = [sol[0]]
    remaining_energy = maximum_energy_of_the_drone
    last_pc = None
    i = 1
    last_pc_used = []
    Res = comp_energy_sol(sol, v0, alpha, beta, wind_vector, wind_speed, distance_unit, grid_w)
    total_energy = round(Res[0], 2)
    while i < len(sol):
        if solution_representation[1][i] == 1:
            last_pc = sol[i]

        tmp.append(sol[i])
        e = comp_energy_sol(tmp, v0, alpha, beta, wind_vector, wind_speed, distance_unit, grid_w)
        remaining_energy -= e[0]

        if (remaining_energy < 0 and last_pc != None and last_pc not in last_pc_used):
            i = sol.index(last_pc)
            solution_representation[2][i] = 1
            solution_representation[3][i] = round(remaining_energy, 2)
            solution_representation[4][i] = 1
            last_pc_used.append(last_pc)
            last_pc = None
            tmp = [sol[i]]

        elif (remaining_energy < 0 and last_pc == None) or (remaining_energy < 0 and last_pc in last_pc_used):
            solution_representation[4][i] = 1
            total_energy = float('inf')
            NPC = float('inf')
            break
        remaining_energy = maximum_energy_of_the_drone
        i = i + 1

    if total_energy == float('inf'):
        NPC = float('inf')
    else:
        NPC = sum(solution_representation[2])

    return sol, total_energy, NPC, solution_representation[2], solution_representation[4], round(Res[1], 2)


def compute_D_max(r, Theta, H):
    return (1 - r) * 2 * H * math.tan(math.radians(Theta / 2))


def compute_altitude(D, r, Theta):
    return D / ((1 - r) * math.tan(math.radians(Theta / 2)))


def back_and_forth(n, m):
    k = 0
    p = 0
    solution = []
    for j in range(n):
        for i in range(m):
            if j % 2 == 0:
                p = (i * n) + j + 1
            else:
                p = (m - i - 1) * n + j + 1
            solution.append(p)
            k = k + 1
    solution.append(1)
    return solution


def back_and_forth_H(n, m):
    k = 0
    p = 0
    solution = []
    for i in range(m):
        for j in range(n):
            if i % 2 == 0:
                p = (i * n) + j + 1
            else:
                p = (i + 1) * n - j
            solution.append(p)
            k = k + 1
    solution.append(1)
    return solution


def back_and_forth_V(n, m):
    k = 0
    p = 0
    solution = []
    for j in range(n):
        for i in range(m):
            if j % 2 == 0:
                p = (i * n) + j + 1
            else:
                p = (m - i - 1) * n + j + 1
            solution.append(p)
            k = k + 1
    solution.append(1)
    return solution


# Function: K_opt
def f_k_opt(sol, charge_points, takeoff,
            maximum_energy_of_the_drone, distance_unit, grid_w, wind_speed,
            v0, wind_vector, alpha, beta):
    sol_tmp = copy.deepcopy(sol)

    ln = len(sol_tmp[0])
    P = random.sample(range(1, ln - 1), 3)  # selecting 3 positions in the vector
    [i, j, k] = np.sort(P)

    # Testing 2-opt
    sol_1 = [[], 1, 1, [], [], 1]
    sol_1[0] = copy.copy(sol_tmp[0])
    sol_1[0][i:j + 1] = list(reversed(sol_1[0][i:j + 1]))
    sol_1 = f_objective(sol_1, charge_points, takeoff, maximum_energy_of_the_drone,
                        distance_unit, grid_w, wind_speed, v0, wind_vector, alpha, beta)

    sol_2 = [[], 1, 1, [], [], 1]
    sol_2[0] = copy.copy(sol_tmp[0])
    sol_2[0][i:k + 1] = list(reversed(sol_2[0][i:k + 1]))
    sol_2 = f_objective(sol_2, charge_points, takeoff, maximum_energy_of_the_drone,
                        distance_unit, grid_w, wind_speed, v0, wind_vector, alpha, beta)

    sol_3 = [[], 1, 1, [], [], 1]
    sol_3[0] = copy.copy(sol_tmp[0])
    sol_3[0][j:k + 1] = list(reversed(sol_3[0][j:k + 1]))
    sol_3 = f_objective(sol_3, charge_points, takeoff, maximum_energy_of_the_drone,
                        distance_unit, grid_w, wind_speed, v0, wind_vector, alpha, beta)

    # Testing 3-opt
    sol_4 = [[], 1, 1, [], [], 1]
    sol_4[0] = copy.copy(sol_tmp[0])
    sol_4[0] = sol_4[0][:i + 1] + list(reversed(sol_4[0][i + 1:j + 1])) + list(reversed(sol_4[0][j + 1:k + 1])) + sol_4[
                                                                                                                      0][
                                                                                                                  k + 1:]
    sol_4 = f_objective(sol_4, charge_points, takeoff, maximum_energy_of_the_drone,
                        distance_unit, grid_w, wind_speed, v0, wind_vector, alpha, beta)

    sol_5 = [[], 1, 1, [], [], 1]
    sol_5[0] = copy.copy(sol_tmp[0])
    sol_5[0] = sol_5[0][:i + 1] + sol_5[0][j + 1:k + 1] + sol_5[0][i + 1:j + 1] + sol_5[0][k + 1:]
    sol_5 = f_objective(sol_5, charge_points, takeoff, maximum_energy_of_the_drone,
                        distance_unit, grid_w, wind_speed, v0, wind_vector, alpha, beta)

    sol_6 = [[], 1, 1, [], [], 1]
    sol_6[0] = copy.copy(sol_tmp[0])
    sol_6[0] = sol_6[0][:i + 1] + list(reversed(sol_6[0][j + 1:k + 1])) + sol_6[0][i + 1:j + 1] + sol_6[0][k + 1:]
    sol_6 = f_objective(sol_6, charge_points, takeoff, maximum_energy_of_the_drone,
                        distance_unit, grid_w, wind_speed, v0, wind_vector, alpha, beta)

    sol_7 = [[], 1, 1, [], [], 1]
    sol_7[0] = copy.copy(sol_tmp[0])
    sol_7[0] = sol_7[0][:i + 1] + sol_7[0][j + 1:k + 1] + list(
        reversed(sol_7[0][i + 1:j + 1])) + sol_7[0][k + 1:]
    sol_7 = f_objective(sol_7, charge_points, takeoff, maximum_energy_of_the_drone,
                        distance_unit, grid_w, wind_speed, v0, wind_vector, alpha, beta)

    if ((sol_1[2] < sol_tmp[2]) or (sol_1[2] == sol_tmp[2] and sol_1[5] < sol_tmp[5])):
        sol_tmp = copy.deepcopy(sol_1)
    elif ((sol_2[2] < sol_tmp[2]) or (sol_2[2] == sol_tmp[2] and sol_2[5] < sol_tmp[5])):
        sol_tmp = copy.deepcopy(sol_2)
    elif ((sol_3[2] < sol_tmp[2]) or (sol_3[2] == sol_tmp[2] and sol_3[5] < sol_tmp[5])):
        sol_tmp = copy.deepcopy(sol_3)
    elif ((sol_4[2] < sol_tmp[2]) or (sol_4[2] == sol_tmp[2] and sol_4[5] < sol_tmp[5])):
        sol_tmp = copy.deepcopy(sol_4)
    elif ((sol_5[2] < sol_tmp[2]) or (sol_5[2] == sol_tmp[2] and sol_5[5] < sol_tmp[5])):
        sol_tmp = copy.deepcopy(sol_5)
    elif ((sol_6[2] < sol_tmp[2]) or (sol_6[2] == sol_tmp[2] and sol_6[5] < sol_tmp[5])):
        sol_tmp = copy.deepcopy(sol_6)
    elif ((sol_7[2] < sol_tmp[2]) or (sol_7[2] == sol_tmp[2] and sol_7[5] < sol_tmp[5])):
        sol_tmp = copy.deepcopy(sol_7)

    return sol_tmp


if __name__ == "__main__":

    # done = False
    is_found_solution = False
    ############################################################################
    method = "_SA"
    user_taille_instances = 0
    altitude_max = 120
    user_T_init = 1
    user_T_min = 0.01
    user_k_init = 100
    user_k_min = 10
    user_execution_number = 20
    v0 = 2.5
    alpha = 5.92
    beta = 231.25
    maximum_energy_3dr_solo = mAh_to_Ah(5200) * 14.8 * 3600
    overlap_ratio = 0.75
    angle_of_view = 94.4

    taille = [8]

    Speed_w = [2, 1]

    V_x = [10, 10, 0.001]
    V_y = [0.001, 11, 10]

    for spw in Speed_w:
        wind_speed = spw
        for siz in taille:
            for coord in range(3):
                user_taille_instances = siz
                grid_w = grid_h = user_taille_instancesbv
                grid = generate_grid(grid_w, grid_h)
                distance_unit = compute_D_max(overlap_ratio, angle_of_view, altitude_max)
                v = get_vector_from(V_x[coord], V_y[coord], distance_unit, grid_w)
                v = get_speed_vector(wind_speed, v)
                wind_vector = v
                wind_vector_normalized = normalize(wind_vector)

                ### Point de décollage entre (grid_w X grid_h)
                landing = 1
                ### Point d'attérissage entre (grid_w X grid_h)
                takeoff = grid_w
                ### Stations de recharge
                charge_points = []
                liste_instances = []

                import os

                fh = open(os.getcwd() + "\instances\instance_" + str(user_taille_instances) + "X" + str(
                    user_taille_instances) + ".txt", "r")
                lines = fh.readlines()
                fh.close()
                for line in lines:
                    line = line.strip()
                    line_ = line[1:len(line) - 1].split(",")
                    for elt in line_:
                        charge_points.append(int(elt.strip()))
                    liste_instances.append(charge_points)
                    charge_points = []

                lines_to_save = []
                instance_index = 0
                for charge_points in liste_instances:
                    print(instance_index + 1)
                    liste_energy = []
                    liste_recharge = []
                    liste_time = []
                    liste_Sol = []
                    liste_RS = []
                    for idx in range(user_execution_number):
                        # done = False
                        T_init = user_T_init  # 0.5
                        T_min = user_T_min  # 0.001
                        T = T_init
                        k_init = user_k_init  # 700
                        k_min = user_k_min  # 0#
                        k = k_init
                        S_0 = [[], float("inf"), float("inf")]
                        S_0[0] = generate_solution(grid_w, grid_h, landing, takeoff, grid)
                        S = f_objective(S_0, charge_points, takeoff, maximum_energy_3dr_solo,
                                        distance_unit, grid_w, wind_speed,
                                        v0, wind_vector, alpha, beta)
                        m = copy.deepcopy(S)
                        Impr = 4
                        while T > T_min:  # or not done:

                            if T > T_min:
                                k = k - 1
                                Sn = f_k_opt(S, charge_points, takeoff, maximum_energy_3dr_solo, distance_unit, grid_w,
                                             wind_speed,
                                             v0, wind_vector, alpha, beta)
                                nNPC = Sn[2]  # Nombre de stations de charge
                                nET = Sn[5]  # Time

                                P = random.uniform(0, 1)
                                P_solution = 0.0
                                try:
                                    P_solution_2 = round(math.exp(-(nET - S[5]) / T), 1)
                                except OverflowError:
                                    P_solution_2 = float('inf')

                                if ((nNPC < S[2]) or (nNPC == S[2] and nET < S[5])) or (nET > S[
                                    5] and P < P_solution_2):  # or (nNPC > e[0] and nET < e[1] and P < P_solution_1):
                                    S = Sn
                                if (S[2] < m[2]) or (S[2] == m[2] and S[5] < m[5]):
                                    m = copy.deepcopy(S)
                                    k = k_init
                                    Impr = 4
                                    is_found_solution = True
                                k -= 1
                                if k < k_min:
                                    T = T * 0.30
                                    k = k_init
                                    Impr -= 1
                                if Impr == 0:
                                    T = T_min

                            if T <= T_min:
                                is_found_solution = False
                                # done = True
                        # print(m)
                        liste_energy.append(str(m[1]))
                        liste_recharge.append(str(m[2]))
                        liste_time.append(str(m[5]))
                        liste_Sol.append(str(m[0]))
                        indexRS = [m[0][ind] for ind, pp in enumerate(m[3]) if pp == 1]
                        liste_RS.append(indexRS)

                    # save result of an instance
                    line_e = ""
                    line_c = ""
                    line_t = ""
                    line_s = ""
                    line_p = ""
                    instance_index += 1
                    for elt in liste_energy:
                        line_e = line_e + ";" + elt
                    line_e = line_e[1:len(line_e)]

                    for elt in liste_recharge:
                        line_c = line_c + ";" + elt
                    line_c = line_c[1:len(line_c)]

                    for elt in liste_time:
                        line_t = line_t + ";" + elt
                    line_t = line_t[1:len(line_t)]

                    for elt in liste_RS:
                        line_pt = ""
                        for item in elt:
                            line_pt = line_pt + "," + str(item)
                        line_pt = line_pt[1:len(line_pt)]
                        line_p = line_p + line_pt + "\n"

                    n1 = user_taille_instances
                    coordinates = [V_x[coord], V_y[coord]]

                    n2 = round(wind_speed, 2)
                    n3 = instance_index
                    fh = open("results/results_" + str(n1) + "X" + str(n1) + method + "_" + str(n2) + "_" + str(
                        user_k_init) + "_" + str(coordinates) + "_" + "_energy.csv", "a")
                    fh.write(line_e + "\n")
                    fh.close()
                    fh = open("results/results_" + str(n1) + "X" + str(n1) + method + "_" + str(n2) + "_" + str(
                        user_k_init) + "_" + str(coordinates) + "_" + "_RS.csv", "a")
                    fh.write(line_c + "\n")
                    fh.close()
                    fh = open("results/results_" + str(n1) + "X" + str(n1) + method + "_" + str(n2) + "_" + str(
                        user_k_init) + "_" + str(coordinates) + "_" + "_time.csv", "a")
                    fh.write(line_t + "\n")
                    fh.close()

                    with open("results/results_" + str(n1) + "X" + str(n1) + method + "_" + str(n2) + "_" + str(
                            user_k_init) + "_" + str(coordinates) + "_" + "_sol.csv", "a") as txt_file:
                        for line in liste_Sol:
                            txt_file.write("".join(line))
                            txt_file.write("\n")
                        txt_file.write("\n \n \n")

                    fh = open("results/results_" + str(n1) + "X" + str(n1) + method + "_" + str(n2) + "_" + str(
                        user_k_init) + "_" + str(coordinates) + "_" + "_indRS.csv", "a")
                    fh.write(line_p + "\n")
                    fh.close()
    # done = True
print("FIN DE LA SIMULATION DE ", len(liste_instances), " INSTANCES!")
