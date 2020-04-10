import math
import random
import numpy as np
################################################################################
# DRAW WITH PYGAME
import pygame
################################################################################

def generate_grid(grid_w, grid_h):
    grid = []
    for i in range(grid_w * grid_h):
        grid.append(i + 1)
    return grid


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


def get_lign_number(num, grid_w):
    if (num / grid_w%1) > 0.0:
        return math.floor(num / grid_w) + 1
    else:
        return math.floor(num / grid_w)


def get_col_number(num, grid_w):
    mod = num%grid_w
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


def generate_solution(grid_w, grid_h, landing_point, takeoff_point, grid):
    solution = []

    for point in grid:
        if point != landing_point and point != takeoff_point:
            solution.append(point)

    random.shuffle(solution)
    solution.insert(0, landing_point)
    solution.insert(len(grid) - 1, takeoff_point)
    # print(solution)

    solution = arrange_solution(solution, grid_w, grid_h, landing_point, takeoff_point)
    return solution


def arrange_solution(solution, grid_w, grid_h, landing_point, takeoff_point):
    for i in range(len(solution) - 1):
        current_point = solution[i]
        next_point = solution[i + 1]
        current_point_neighbours = neighbours(grid_w, grid_h, current_point)
        # print(current_point_neighbours)
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


def compute_battery_energy_Ah(U, C):
    return U * C


def compute_battery_energy_mAh(U, C):
    return U * mAh_to_Ah(C)


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
    X = (V_xy[0]) / m
    Y = (V_xy[1]) / m
    return [X, Y]


def det(A, B):
    return A[0] * B[1] - B[0] * A[1]


def get_speed_vector(speed=0, V_xy=[0, 0]):
    X = 0
    Y = 0
    m = get_vector_magnitude(V_xy)
    if m > 0:
        X = (speed * V_xy[0]) / m
        Y = (speed * V_xy[1]) / m

    return [X, Y]


def get_scalar_product(A, B):
    return A[0] * B[0] + A[1] * B[1]


def get_distance_between(A, B, Distance_unit, grid_w):
    P = get_vector_from(A, B, Distance_unit, grid_w)
    return math.sqrt(P[0] ** 2 + P[1] ** 2)


def compute_power_energy_time(betas, x, y, distance_Unit, grid_w, speed,
                              acceleration,
                              speed_z, acceleration_z, weight_payload,
                              wind_vector):
    vector_from_x_y = get_vector_from(x, y, distance_Unit, grid_w)
    speed_vector = get_speed_vector(speed, vector_from_x_y)
    d = get_distance_between(x, y, distance_unit, grid_w)
    t = d / speed

    # print("Distance:", d, "m")
    # print("Time:", t, "s")

    M_beta_1 = np.array([[betas[0]], [betas[1]], [betas[2]]])
    M_beta_2 = np.array([[betas[3]], [betas[4]], [betas[5]]])
    M_beta_3 = np.array([[betas[6]], [betas[7]], [betas[8]]])

    M1 = np.array([[speed], [acceleration], [speed * acceleration]])
    M2 = np.array([[speed_z], [acceleration_z], [speed_z * acceleration_z]])
    M3 = np.array(
        [[weight_payload], [get_scalar_product(speed_vector, wind_vector)],
         [1]])

    P = M_beta_1.transpose() @ M1 + M_beta_2.transpose() @ M2 + M_beta_3.transpose() @ M3
    return P[0][0], (P * ((t / 60) / 60))[0][0], t


def compute_energy(solution, charge_points, betas, distance_Unit, grid_w, speed,
                   acceleration, speed_z, acceleration_z, weight_payload,
                   wind_vector):
    energy = 0
    power = 0
    time = 0
    n = len(solution)
    acc = acceleration

    for i in range(n - 1):
        decceleration = False
        if solution[i + 1] in charge_points or i + 1 == n - 1:
            decceleration = True
            d = get_distance_between(solution[i], solution[i + 1],
                                     distance_unit, grid_w)
            acceleration = -speed / d
            # if i == n-2:
            #     print(solution[i+1], "is charge point with", acceleration,
            # "m/s2", "on distance", d, "with speed", speed, "m/s")
        else:
            acceleration = acc

        power_energy_time = compute_power_energy_time(betas, solution[i],
                                                      solution[i + 1],
                                                      distance_Unit, grid_w,
                                                      speed, acceleration,
                                                      speed_z, acceleration_z,
                                                      weight_payload,
                                                      wind_vector)
        power += power_energy_time[0]
        energy += power_energy_time[1]
        time += power_energy_time[2]

    """print("Drone 3DR SOLO Energy:", round(mAh_to_Ah(5200)*14.8, 2), "Wh")
    print("total power:", round(power, 2), "W")
    print("total energy:", round(energy, 2), "Wh")
    print("total time:", round(time, 2), "seconds")
    print("total time:", round(time/60, 2), "minutes")
    print("total time:", round((time/60)/60, 2), "hours")"""
    return energy

def is_feasible(solution, charge_points, takeoff_point,
                maximum_energy_of_the_drone, betas, distance_unit, grid_w,
                speed, acceleration, speed_z, acceleration_z, weight_payload,
                wind_vector):
    return True
    sub_path = []
    sub_paths = []
    consumed_energy = 0
    remaining_energy_at_charge_point = maximum_energy_of_the_drone
    remaining_energies_at_charge_points = 0
    remaining_energies_at_charge_points_list = []
    for point in solution:
        sub_path.append(point)
        if point in charge_points or point == takeoff_point:
            sub_paths.append(sub_path)
            energy_at_charge_point = compute_energy(sub_path, charge_points,
                                                    betas, distance_unit,
                                                    grid_w, speed, acceleration,
                                                    speed_z, acceleration_z,
                                                    weight_payload, wind_vector)
            consumed_energy += energy_at_charge_point
            remaining_energy_at_charge_point -= energy_at_charge_point

            if remaining_energy_at_charge_point < 0:
                return False

            if abs(remaining_energy_at_charge_point) > maximum_energy_of_the_drone:
                remaining_energy_at_charge_point = maximum_energy_of_the_drone

            remaining_energies_at_charge_points += abs(remaining_energy_at_charge_point)
            remaining_energies_at_charge_points_list.append(remaining_energy_at_charge_point)

            if point in charge_points:
                percent = round((remaining_energy_at_charge_point / maximum_energy_of_the_drone) * 100)
                """if percent < 0:
                    percent = 0.0"""
                # print(round(remaining_energy_at_charge_point, 2), "Wh at point", point, "| remainig energy ", percent, "%")
                remaining_energy_at_charge_point = maximum_energy_of_the_drone
            sub_path = []
            sub_path.append(point)
    return True


### Fonction objective
def f_objective(solution, charge_points, takeoff_point,
                maximum_energy_of_the_drone, betas, distance_unit, grid_w,
                speed, acceleration, speed_z, acceleration_z, weight_payload,
                wind_vector):

    pc_solution = []
    pc_used = []
    energies = []
    for elt in solution:
        if elt in charge_points:
            pc_solution.append(1)
        else:
            pc_solution.append(0)
        pc_used.append(0)
        energies.append(0)

    # representation de la solution
    solution_representation = [
        solution,
        pc_solution,
        pc_used,
        energies,
    ]
    tmp = [solution[0]]
    remaining_energy = maximum_energy_of_the_drone
    last_pc = None
    i = 1
    k_max = 200
    k = k_max
    factor = 0.01
    last_pc_used = []
    total_energy = compute_energy(solution, charge_points, betas, distance_unit,
                                  grid_w, speed, acceleration, speed_z,
                                  acceleration_z, weight_payload, wind_vector)
    while i < len(solution):
        if solution_representation[1][i] == 1:
            last_pc = solution[i]


        tmp.append(solution[i])
        e = compute_energy(tmp, charge_points, betas, distance_unit,
                                  grid_w, speed, acceleration, speed_z,
                                  acceleration_z, weight_payload, wind_vector)
        remaining_energy -= e

        # print("tmp:", tmp)
        # print("tmp energy:", e, "wh")
        # print("remaining energy:", remaining_energy)
        # print("maximum energy of the drone:", mAh_to_Ah(5200) * 14.8, "\n")

        if (remaining_energy < 0 and last_pc != None and last_pc not in last_pc_used):
            i = solution.index(last_pc)
            solution_representation[2][i] = 1
            solution_representation[3][i] = round(remaining_energy, 2)
            last_pc_used.append(last_pc)
            last_pc = None
            tmp = [solution[i]]
        elif (remaining_energy < 0 and last_pc == None) or (remaining_energy < 0 and last_pc in last_pc_used):
            total_energy = float('inf')
            NPC = float('inf')
            break
        remaining_energy = maximum_energy_of_the_drone
        i = i + 1

    # print(solution_representation[0])
    # print(solution_representation[1])
    # print(solution_representation[2])

    if total_energy == float('inf'):
        NPC = float('inf')
    else:
        NPC = sum(solution_representation[2])


    return NPC, total_energy, solution_representation

# Solution voisine
def neighbours_solution(s, grid_w, grid_h, takeoff_point, landing_point):
    solu = []
    for i in range(len(s)):
        solu.append(s[i])
    solu.remove(takeoff_point)
    solu.remove(landing_point)

    max_distance = 0
    max_i = 0
    """for i in range(len(solu) - 1):
        d = get_distance_between(solu[i], solu[i + 1], distance_unit, grid_w)
        if d > max_distance:
            max_distance = d
            max_i = i
    solu[max_i], solu[max_i + 1] = solu[max_i + 1], solu[max_i]"""

    if round(random.uniform(0, 1), 1) < 0:
        max_distance = 0
        max_i = 0
        for i in range(len(solu)-1):
            d = get_distance_between(solu[i], solu[i+1], distance_unit, grid_w)
            if d > max_distance:
                max_distance = d
                max_i = i
        solu[max_i], solu[max_i+1] = solu[max_i+1], solu[max_i]

    else:
        j = random.randrange(0, len(solu) - 1)
        nghbrs_j = neighbours(grid_w, grid_h, solu[j])

        k = random.randrange(0, len(solu) - 1)
        nghbrs_k = neighbours(grid_w, grid_h, solu[k])

        if takeoff_point in nghbrs_j:
            nghbrs_j.remove(takeoff_point)
        if landing_point in nghbrs_j:
            nghbrs_j.remove(landing_point)
        if len(nghbrs_j) > 0:
            rand_point = random.choice(nghbrs_j)
            idx_rand_point = solu.index(rand_point)
            solu[j], solu[idx_rand_point] = solu[idx_rand_point], solu[j]

        if takeoff_point in nghbrs_k:
            nghbrs_k.remove(takeoff_point)
        if landing_point in nghbrs_k:
            nghbrs_k.remove(landing_point)
        if len(nghbrs_k) > 0:
            rand_point = random.choice(nghbrs_k)
            idx_rand_point = solu.index(rand_point)
            solu[k], solu[idx_rand_point] = solu[idx_rand_point], solu[k]

    # print("solu", solu)
    solu.append(takeoff_point)
    solu.insert(0, landing_point)
    # print("solu", solu)
    return solu


def compute_D_max(r, Theta, H):
    return (1-r)*2*H*math.tan(math.radians(Theta/2))

def compute_altitude(D, r, Theta):
    return D/((1-r)*math.tan(math.radians(Theta/2)))

def back_and_forth(n, m):
    k = 0
    p = 0
    solution = []
    for j in range(n):
        for i in range(m):
            if j%2 == 0:
                p = (i*n) + j + 1
            else:
                p = (m-i-1)*n + j + 1
            solution.append(p)
            k = k + 1
    return solution
            

################################################################################
# DRAW WITH PYGAME
def draw_solution(path, screen, color, w):
    for i in range(len(path)-1):
        num_start = path[i]
        num_end = path[i + 1]
        pos_start = points[num_start]
        pos_end = points[num_end]
        pygame.draw.line(screen, color, pos_start, pos_end, w)
        
        endX, endY = pos_end
        startX, startY = pos_start
        dX = endX - startX
        dY = endY - startY

        # vector length
        Len = math.sqrt(dX**2 + dY**2) # use Hypot if available

        # normalized direction vector components
        udX = dX / Len
        udY = dY / Len

        # perpendicular vector
        perpX = -udY
        perpY = udX

        # points forming arrowhead
        # with length L and half-width H
        arrowend = (pos_end)
        L = 6
        H = 4
        leftX = endX - L * udX + H * perpX
        leftY = endY - L * udY + H * perpY
        left = (leftX, leftY)

        rightX = endX - L * udX - H * perpX
        rightY = endY - L * udY - H * perpY
        right = (rightX, rightY)

        # pygame.draw.line(screen, color, pos_end, left, w)
        # pygame.draw.line(screen, color, pos_end, right, w)
        # pygame.draw.line(screen, color, left, right, w)
        pygame.draw.polygon(screen, color, [pos_end, left, right, pos_end], 0)
#################################################################################

if __name__ == "__main__":

    # dimensions de la map
    grid_w = 4
    grid_h = 4
    ############################################################################
    # DRAW WITH PYGAME
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    CELL_WIDTH = 50
    CELL_HEIGHT = 50
    GRID_SIZE_W = 0
    GRID_SIZE_H = 0
    MARGIN = 1
    pygame.init()
    pygame.display.set_caption("Visualisation")
    done = False
    clock = pygame.time.Clock()
    GRID_SIZE_W = grid_w
    GRID_SIZE_H = grid_h
    SCREEN_WIDTH = CELL_WIDTH * GRID_SIZE_W + MARGIN * GRID_SIZE_W + MARGIN + 350
    SCREEN_HEIGHT = CELL_HEIGHT * GRID_SIZE_H + MARGIN * GRID_SIZE_H + MARGIN
    WINDOW_SIZE = [SCREEN_WIDTH, SCREEN_HEIGHT]
    screen = None #pygame.display.set_mode(WINDOW_SIZE)
    font = pygame.font.SysFont("Arial", 10)
    # end_sound_effect = pygame.mixer.Sound('to-the-point.ogg')
    is_found_solution = False
    ############################################################################

    ############################################################################
    ############################################################################

    
    user_taille_instances = 0
    user_instance_number = 0
    user_nb_points_recharge = 0
    user_execution_number = 0
    user_landing = 0
    user_takeoff = 0
    user_charge_points = []
    user_overlap_ratio = 0.75
    user_altitude_max = 50
    user_angle_of_view = 94.4
    user_distance_unit = 0
    user_T_init = 0
    user_T_min = 0
    user_k_init = 0
    user_k_min = 0
    
    menu = True
    while menu:
        print("\n##########################################################")
        print("################ [ SIM ] BF - SA - BFSA ##################")
        print("##########################################################")
        print("##### 1- Générer les instances")
        print("##### 2- Simuler avec BF")
        print("##### 3- Simuler avec SA")
        print("##### 4- Simuler avec BFSA \n")

        user_choice = input("Faire un choix @> ")

        if(user_choice == "1"):
            user_instance_number = int(input("Entrer le nombre d'instances à générer @> "))
            user_taille_instances = int(input("Entrer la taille des instances à générer @> "))
            user_nb_points_recharge = int(input("Entrer le nombre max de stations de recharges @> "))

            instances_lines = []
            
            import os
            fh = open("instances/instance_"+ str(user_taille_instances) + "X" + str(user_taille_instances) + ".txt", "w+")
            for j in range(user_instance_number):
                tmp_charge_points = []
                p = random.randrange(user_taille_instances * user_taille_instances)
                for i in range(user_nb_points_recharge):
                    while p in tmp_charge_points:
                        p = random.randrange(user_taille_instances * user_taille_instances)
                    tmp_charge_points.append(p)
                instances_lines.append(str(tmp_charge_points))
                fh.write(str(tmp_charge_points) + "\n")
            fh.close()

            
            """import os
            print("INSTANCES ", instances_lines)
            fh = open("instances/instance_"+ str(user_taille_instances) + "X" + str(user_taille_instances) + ".txt", "a")
            fh.writelines(instances_lines)
            fh.close()"""
            print("Génération des instances terminé!")
        

        elif(user_choice == "2" or user_choice == "3" or user_choice == "4"):
            user_taille_instances = int(input("Taille des instances à simuler @> "))
            user_altitude_max = float(input("Altitude max du drone (120m) @> "))
            user_overlap_ratio = float(input("Taux de chevauchement (0.75) @> "))
            user_angle_of_view = float(input("Angle de vue camera (94.4°) @> "))
            user_T_init = float(input("Temperature initiale (0.5) @> ")) 
            user_T_min = float(input("Temperature minimale (0.001) @> ")) 
            user_k_init = float(input("k initial (700) @> ")) 
            user_k_min = float(input("K minimum (0) @> "))
            user_execution_number = int(input("Entrer le nombre d'éxecutions de chaque instance @> "))
            menu = False

    ############################################################################
    ############################################################################

    ### Point de décollage entre (grid_w X grid_h)
    landing = 1
    ### Point d'attérissage entre (grid_w X grid_h)
    takeoff = grid_w
    ### Stations de recharge
    charge_points = []
    liste_instances = []
    GRID_SIZE_W = GRID_SIZE_H= grid_w = grid_h = user_taille_instances
    SCREEN_WIDTH = CELL_WIDTH * GRID_SIZE_W + MARGIN * GRID_SIZE_W + MARGIN + 350
    SCREEN_HEIGHT = CELL_HEIGHT * GRID_SIZE_H + MARGIN * GRID_SIZE_H + MARGIN
    WINDOW_SIZE = [SCREEN_WIDTH, SCREEN_HEIGHT]
    screen = pygame.display.set_mode(WINDOW_SIZE)

    ### #
    import os
    fh = open(os.getcwd() + "\instances\instance_"+ str(user_taille_instances) + "X" + str(user_taille_instances) + ".txt", "r")
    lines = fh.readlines()
    fh.close()
    for line in lines:
        line = line.strip()
        line_ = line[1:len(line)-1].split(",")
        for elt in line_:
            charge_points.append(int(elt.strip()))
        liste_instances.append(charge_points)
        charge_points = []
    ### #

    grid = generate_grid(grid_w, grid_h)
    weight_payload = 0.0  # g
    weight_3dr_solo = 1.8  # kg 1.8  # g
    width_3dr_solo = 0.25  # m # 0.00025 #km‬  #
    maximum_energy_3dr_solo = mAh_to_Ah(5200) * 14.8 # Energie max du drone
    overlap_ratio = user_overlap_ratio #0.75 # % # Overlap ratio, taux de chevauchement
    angle_of_view = user_angle_of_view #94.4 # ° # Angle de vue de la caméra
    distance_unit = compute_D_max(overlap_ratio, angle_of_view, user_altitude_max) # 40  # m
    altitude_max = user_altitude_max#compute_altitude(distance_unit, overlap_ratio, angle_of_view) # m
    # print("altitude", altitude_max)
    # print("distance unit", distance_unit)
    speed_z = 0  # m/s # km/h
    acceleration_z = 0  # m/s2 # km/h
    speed_optimum_3dr_solo = 0.5  # m/s #  1.8 # km/h Vopt= Vitesse optimale du drone 3DR Solo
    speed = speed_optimum_3dr_solo
    speed_xy = [0, 0]
    wind_speed = 3.88  # m/s # 14 # km/h # Vitesse du vent
    v = get_vector_from(1, 6, distance_unit, grid_w)
    v = get_speed_vector(wind_speed, v)
    wind_vector = v
    wind_vector_normalized = normalize(wind_vector)
    # Coefficients pour l'Equation de Tseng Ming
    betas_3dr_solo = [-1.526, 3.934, 0.968, 18.125, 96.613, -1.085, 0.220, 1.332, 433.9]
    betas_dji = [-2.595, 0.116, 0.824, 18.321, 31.745, 13.282, 0.197, 1.43, 251.7]
    accelerations = [4, 6, 12, 40, 0, -4]

    lines_to_save = []
    instance_index = 1
    for charge_points in liste_instances:
        liste_energy = []
        liste_recharge = []
        for idx in range(user_execution_number):
            done = False
            T_init = user_T_init #0.5
            T_min = user_T_min #0.001
            T = T_init
            k_init = user_k_init # 700
            k_min = user_k_min#0#
            k = k_init
            S_0 = generate_solution(grid_w, grid_h, landing, takeoff, grid)
            if user_choice == "1" or user_choice == "4":
                S_0 = back_and_forth(grid_w, grid_h)
            S = S_0
            G = S_0
            # print(S)
            e = f_objective(S, charge_points, takeoff, maximum_energy_3dr_solo,
                            betas_3dr_solo, distance_unit, grid_w, speed,
                            accelerations[4], 0, 0, weight_payload, wind_vector)
            m = e
            """while not is_feasible(S, charge_points, takeoff, maximum_energy_3dr_solo,
                                betas_3dr_solo, distance_unit, grid_w, speed,
                                accelerations[4], 0, 0, weight_payload, wind_vector):
                # print(S, " Drone Can't reach recharging point")
                negative_path = False
                S = generate_solution(grid_w, grid_h, landing, takeoff, grid)
                e = f_objective(S, charge_points, takeoff, maximum_energy_3dr_solo,
                                betas_3dr_solo, distance_unit, grid_w, speed,
                                accelerations[4], 0, 0, weight_payload, wind_vector)"""

            """print("==First Solution", "=" * 255)
            print(round(m[1], 2), "Wh to complete and", m[0], "Recharging Points")
            print(G)
            print("Recharging Points:", m[2][1])
            print("Recharging Points used:", m[2][2])
            print("Energy after this Recharging Points used:", m[2][3])
            print("=" * 255, "\n")"""

            if user_choice == "3" or user_choice == "4":
                while T > T_min or not done:
                    ########################################################################
                    # DRAW WITH PYGAME
                    if done == True:
                        break

                    charge_points_coords = []
                    for cp in charge_points:
                        charge_points_coords.append((get_col_number(cp, GRID_SIZE_W), get_lign_number(cp, GRID_SIZE_W)))
                        # print(cp, (get_col_number(cp, grid_w), get_lign_number(cp, grid_w)))
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True

                    used_charge_points = []
                    solution_representation = m[2]
                    for i in range(len(solution_representation[0])):
                        if solution_representation[2][i] == 1:
                            used = solution_representation[0][i]
                            used_charge_points.append((get_col_number(used, GRID_SIZE_W), get_lign_number(used, GRID_SIZE_W)))
                    # print(used_charge_points)

                    screen.fill(BLACK)

                    # Draw the grid
                    num = GRID_SIZE_W * GRID_SIZE_H

                    points = {}
                    tmp_points = []
                    for row in range(GRID_SIZE_H):
                        for column in range(GRID_SIZE_W):
                            color = (155, 155, 155)
                            x = (MARGIN + CELL_WIDTH) * column + MARGIN
                            y = (MARGIN + CELL_HEIGHT) * row + MARGIN
                            # print([x, y, CELL_WIDTH, CELL_HEIGHT])
                            # if first_time:
                            # print(column+1, abs(row-GRID_SIZE_H))
                            tmp_points.append((x + CELL_WIDTH / 2, y + CELL_HEIGHT / 2))
                            # tmp_points.append((column+1, abs(row-GRID_SIZE_H)))
                            pygame.draw.rect(screen, color, [x, y, CELL_WIDTH, CELL_HEIGHT])
                            if (column+1, abs(row-GRID_SIZE_H)) in charge_points_coords:
                                pygame.draw.rect(screen, (255, 100, 100), [x, y, CELL_WIDTH, CELL_HEIGHT])
                            if (column+1, abs(row-GRID_SIZE_H)) in used_charge_points:
                                pygame.draw.rect(screen, (255, 255, 255), [x, y, CELL_WIDTH, CELL_HEIGHT])

                            # pygame.draw.rect(screen, color, [x, y, CELL_WIDTH, CELL_HEIGHT])
                        # if first_time:
                        for p in tmp_points[::-1]:
                            points[num] = p
                            num = num - 1
                        tmp_points = []
                    ########################################################################

                    if T > T_min:
                        k = k - 1
                        Sn = neighbours_solution(S, grid_w, grid_h, takeoff, landing)
                        en = f_objective(Sn, charge_points, takeoff,
                                        maximum_energy_3dr_solo,
                                        betas_3dr_solo, distance_unit, grid_w, speed,
                                        accelerations[4], 0, 0, weight_payload,
                                        wind_vector)

                        """while not is_feasible(Sn, charge_points, takeoff,
                                            maximum_energy_3dr_solo,
                                            betas_3dr_solo, distance_unit, grid_w, speed,
                                            accelerations[4], 0, 0, weight_payload,
                                            wind_vector):
                            # print(Sn, " Drone Can't reach recharging point")
                            negative_path = False
                            Sn = neighbours_solution(S, grid_w, grid_h, takeoff, landing)
                            en = f_objective(Sn, charge_points, takeoff,
                                            maximum_energy_3dr_solo,
                                            betas_3dr_solo, distance_unit, grid_w, speed,
                                            accelerations[4], 0, 0, weight_payload,
                                            wind_vector)"""

                        nNPC = en[0] # Nombre de stations de charge
                        nET = en[1] # Energie
                        # print("diffE:", diffE)

                        P = random.uniform(0, 1)
                        P_solution = 0.0
                        try:
                            P_solution_1 = round(math.exp(-(nNPC - e[0]) / T), 1)
                            P_solution_2 = round(math.exp(-(nET - e[1]) / T), 1)
                        except OverflowError:
                            P_solution_1 = float('inf')
                            P_solution_2 = float('inf')



                        if ((nNPC < e[0]) or (nNPC == e[0] and nET < e[1])) or (nNPC == e[0] and nET > e[1] and P < P_solution_2): # or (nNPC > e[0] and nET < e[1] and P < P_solution_1):
                            S = Sn
                            e = en

                        if (e[0] < m[0]) or (e[0] == m[0] and e[1] < m[1]):
                            G = S
                            m = e
                            k = k_init

                            """print("==Chooosed solution", "="*255)
                            print(round(m[1], 2), "Wh to complete and", m[0], "Recharging Points")
                            print(G)
                            print("Recharging Points:", solution_representation[1])
                            print("Recharging Points used:", solution_representation[2])
                            print("Energy after this Recharging Points used:", solution_representation[3])
                            print("="*255, "\n")"""
                            is_found_solution = True
                        k -= 1
                        if k < k_min:
                            T = T*0.70
                            k = k_init

                    ########################################################################
                    # DRAW WITH PYGAME
                    if T < T_min:
                        # end_sound_effect.play()
                        is_found_solution = False
                        done = True
                    if T > T_min:
                        draw_solution(en[2][0], screen, RED, 15)
                        draw_solution(e[2][0], screen, WHITE, 6)
                    draw_solution(m[2][0], screen, GREEN, 3)

                    label_temp = font.render("Temperature: " + str(round(T, 4)), True, (255, 255, 255))
                    label_equi = font.render("Equilibrium: " + str(k), 1, (255, 255, 255))
                    label_ener = font.render("Consumed Energy: " + str(round(m[1], 2)) + "Wh", True, (255, 255, 255))
                    label_rech = font.render("Recharging Points: " + str(m[0]), True, (255, 255, 255))
                    label_dist = font.render("Distance Unit: " + str(distance_unit) + "m", True, (255, 255, 255))
                    label_spee = font.render("Optimum Speed: " + str(speed_optimum_3dr_solo) + "m/s", True, (255, 255, 255))
                    label_dren = font.render("Maximum Drone Energy: " + str(round(maximum_energy_3dr_solo, 2)) + "Wh", True, (255, 255, 255))
                    label_wspe = font.render("Wind Speed: " + str(wind_speed) + "m/s", True, (255, 255, 255))
                    label_wdir = font.render("Wind Direction: (" + str(round(wind_vector_normalized[0], 2)) + ", " + str(round(wind_vector_normalized[1], 2)) + ")", True, (255, 255, 255))
                    label_orat = font.render("Overlap Ratio:" + str(overlap_ratio), True, (255, 255, 255))
                    label_aovi = font.render("Angle of View:" + str(angle_of_view), True, (255, 255, 255))
                    label_alti = font.render("Altitude:" + str(altitude_max), True, (255, 255, 255))
                    label_sepr = font.render("==================", True, (255, 255, 255))
                    label_instance_index = font.render("INSTANCE NUMBER: " + str(instance_index) + "/" + str(len(liste_instances)), True, (255, 255, 255))
                    label_execution_num = font.render("EXECUTION NUMBER: " + str(idx+1) + "/" + str(user_execution_number), True, (255, 255, 255))
                    MARGIN_TEXT = 10
                    screen.blit(label_temp, (SCREEN_WIDTH - 345, MARGIN_TEXT))
                    screen.blit(label_equi, (SCREEN_WIDTH - 345, MARGIN_TEXT*2))
                    screen.blit(label_ener, (SCREEN_WIDTH - 345, MARGIN_TEXT*3))
                    screen.blit(label_rech, (SCREEN_WIDTH - 345, MARGIN_TEXT*4))
                    screen.blit(label_dist, (SCREEN_WIDTH - 345, MARGIN_TEXT*5))
                    screen.blit(label_spee, (SCREEN_WIDTH - 345, MARGIN_TEXT*6))
                    screen.blit(label_dren, (SCREEN_WIDTH - 345, MARGIN_TEXT*7))
                    screen.blit(label_wspe, (SCREEN_WIDTH - 345, MARGIN_TEXT*8))
                    screen.blit(label_wdir, (SCREEN_WIDTH - 345, MARGIN_TEXT*9))
                    screen.blit(label_orat, (SCREEN_WIDTH - 345, MARGIN_TEXT*10))
                    screen.blit(label_aovi, (SCREEN_WIDTH - 345, MARGIN_TEXT*11))
                    screen.blit(label_alti, (SCREEN_WIDTH - 345, MARGIN_TEXT*12))
                    screen.blit(label_sepr, (SCREEN_WIDTH - 345, MARGIN_TEXT*13))
                    screen.blit(label_instance_index, (SCREEN_WIDTH - 345, MARGIN_TEXT*14))
                    screen.blit(label_execution_num, (SCREEN_WIDTH - 345, MARGIN_TEXT*15))
                    # screen.blit(label_rech, (SCREEN_WIDTH - 300, 260))
                    clock.tick(30)
                    pygame.display.flip()
                    ########################################################################
                """print("==Final Solution", "=" * 255)
                print(round(m[1], 2), "Wh to complete and", m[0], "Recharging Points")
                print(G)
                print("Recharging Points:", solution_representation[1])
                print("Recharging Points used:", solution_representation[2])
                print("Energy after this Recharging Points used:", solution_representation[3])
                print("=" * 255, "\n")"""
                liste_energy.append(str(m[1]))
                liste_recharge.append(str(m[0]))
                print(liste_energy)
            else:
                liste_energy.append(str(m[1]))
                liste_recharge.append(str(m[0]))
        
        # save result of an instance
        line = ""
        line_c = ""
        for elt in liste_energy:
            line = line + ";" + elt
        line = line[1:len(line)]
        lines_to_save.append(line)
        instance_index += 1

        for elt in liste_recharge:
            line_c = line_c + ";" + elt
        line_c = line_c[1:len(line)]

        print(line)
        print(line_c)
        
        method = ""
        if user_choice == "2":
            method = "_BF"
        elif user_choice == "3":
            method = "_SA"
        elif user_choice == "4":
            method = "_BFSA"
            
        fh = open("results/results_"+ str(user_taille_instances) + "X" + str(user_taille_instances) + method + "_energy.txt", "a")
        fh.write(line + "\n")
        fh.close()
        fh = open("results/results_"+ str(user_taille_instances) + "X" + str(user_taille_instances) + method + "_chargings.txt", "a")
        fh.write(line_c + "\n")
        fh.close()
    done = True
    pygame.quit()
print("FIN DE LA SIMULATION DE ", len(liste_instances), " INSTANCES!")
    
