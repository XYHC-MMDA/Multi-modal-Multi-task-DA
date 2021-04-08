class raw_classes:
    class_list = [
        # totol [1, 23]: 78942623
        ('noise', 2061156),  # 0
        ('animal', 5385),  # 1
        ('human.pedestrian.adult', 2156470),  # 2
        ('human.pedestrian.child', 9655),  # 3
        ('human.pedestrian.construction_worker', 139443),  # 4
        ('human.pedestrian.personal_mobility', 8723),  # 5
        ('human.pedestrian.police_officer', 9159),  # 6
        ('human.pedestrian.stroller', 8809),  # 7
        ('human.pedestrian.wheelchair', 12168),  # 8
        ('movable.object.barrier', 9305106),  # 9
        ('movable.object.debris', 66861),  # 10
        ('movable.object.pushable_pullable', 718641),  # 11
        ('movable.object.trafficcone', 736239),  # 12
        ('static.object.bicycle_rack', 163126),  # 13
        ('vehicle.bicycle', 141351),  # 14
        ('vehicle.bus.bendy', 357463),  # 15
        ('vehicle.bus.rigid', 4247297),  # 16
        ('vehicle.car', 38104219),  # 17
        ('vehicle.construction', 1514414),  # 18
        ('vehicle.emergency.ambulance', 2218),  # 19
        ('vehicle.emergency.police', 59590),  # 20
        ('vehicle.motorcycle', 427391),  # 21
        ('vehicle.trailer', 4907511),  # 22
        ('vehicle.truck', 15841384),  # 23

        # total {0, 24, ...31}: 1106713569
        ('flat.driveable_surface', 316958899),  # 24
        ('flat.other', 8559216),  # 25
        ('flat.sidewalk', 70197461),  # 26
        ('flat.terrain', 70289730),  # 27
        ('static.manmade', 178178063),  # 28
        ('static.other', 817150),  # 29
        ('static.vegetation', 122581273),  # 30
        ('vehicle.ego', 337070621),  # 31
    ]

    # total points
    num_pts = sum([k[1] for k in class_list])


class LidarSegChallenge:
    class_list = [
        ('ignore', [1, 5, 7, 8, 10, 11, 13, 19, 20, 0, 29, 31]),  # 0
        ('barrier', [9]),  # 1
        ('bicycle', [14]),  # 2
        ('bus', [15, 16]),  # 3
        ('car', [17]),  # 4
        ('construction_\nvehicle', [18]),  # 5
        ('motorcycle', [21]),  # 6
        ('pedestrian', [2, 3, 4, 6]),  # 7
        ('traffic.cone', [12]),  # 8
        ('trailer', [22]),  # 9
        ('truck', [23]),  # 10
        ('driveable_\nsurface', [24]),  # 11
        ('other.flat', [25]),  # 12
        ('sidewalk', [26]),  # 13
        ('terrain', [27]),  # 14
        ('manmade', [28]),  # 15
        ('vegetation', [30])  # 16
    ]

    # fig_size = (16, 8)
    fig_size = (14, 4)
    bar_colors = ['grey'] * len(class_list)
    for i in range(1, 11):
        bar_colors[i] = 'skyblue'
    for i in range(11, 17):
        bar_colors[i] = 'darkgoldenrod'


class xMUDA:
    class_list = [
        ('vehicle', [15, 16, 17, 18, 22, 23]),  # 0
        ('pedestrian', [2, 3, 4, 6]),  # 1
        ('bike', [14, 21]),  # 2
        ('traffic_\nboundary', [9, 12]),  # 3
        ('background', [1, 5, 7, 8, 10, 11, 13, 19, 20, 0, 29, 31, 24, 25, 26, 27, 28, 30])  # 4
    ]

    fig_size = (5, 4)
    bar_colors = ['skyblue'] * len(class_list)
    bar_colors[-1] = 'darkgoldenrod'


class Contrast:
    class_list = [
        ('vehicle', [15, 16, 17, 18, 22, 23]),  # 0
        ('pedestrian', [2, 3, 4, 6]),  # 1
        ('bike', [14, 21]),  # 2
        ('traffic_\nboundary', [9, 12]),  # 3
        ('driveable_\nsurface', [24]),  # 4
        ('other_flat', [25]),  # 5
        ('sidewalk', [26]),  # 6
        ('terrain', [27]),  # 7
        ('manmade', [28]),  # 8
        ('vegetation', [30]),  # 9
        ('ignore', [1, 5, 7, 8, 10, 11, 13, 19, 20, 0, 29, 31])  # 10
    ]

    fig_size = (11, 4)
    bar_colors = ['skyblue'] * len(class_list)
    for i in range(4, 10):
        bar_colors[i] = 'darkgoldenrod'
    bar_colors[-1] = 'grey'

