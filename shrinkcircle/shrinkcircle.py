import math
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

class Circle():
    def __init__(self, x: float, y: float, radius:float):
        self.x = x
        self.y = y
        self.radius = radius

    def get_center(self):
        return self.x, self.y
    
    def get_radius(self) -> float:
        return self.radius

    def area(self) -> float:
        return math.pi * self.radius * self.radius

    def circumference(self) -> float:
        return 2 * math.pi * self.radius
    
    def is_contained(self, x, y) -> bool:
        return math.sqrt((x - self.x)**2 + (y - self.y)**2) <= self.radius
    
    def convert_xy_to_polar_coords(self, x, y):
        r = math.sqrt(x**2 + y**2)
        theta = math.atan2(y, x)
        return r, theta

    def convert_polar_coords_to_xy(self, r, theta):
        x = self.x + r * math.cos(theta)
        y = self.y + r * math.sin(theta)
        return x, y
    
    def generate_random_point_in_circle(self, distribution="uniform", random_seed=0):
        assert distribution in ["uniform", "normal"]
        random.seed(random_seed)
        if distribution == "uniform":
            r = random.uniform(0, self.radius)
            theta = random.uniform(0, 2*math.pi)
        x, y = self.convert_polar_coords_to_xy(r, theta)
        return x, y
    
    def plot(self, **kwargs):
        fig = plt.figure()
        ax = plt.axes()
        c = patches.Circle(xy=(self.x, self.y), radius=self.radius, **kwargs)
        ax.add_patch(c)
        plt.axis('scaled')
        ax.set_aspect('equal')
        plt.show()

    
class ShrinkCircle():
    def __init__(self):
        self.init_circle = Circle(0, 0, 1)
        self.current_circle = Circle(self.init_circle.x, self.init_circle.y, self.init_circle.radius)
        self.init_shrink_ratio = 1
        self.current_shrink_ratio = 1
        self.target_point = self.generate_target_point_randomly()

    def get_center(self) -> tuple:
        return self.current_circle.x, self.current_circle.y
    
    def get_radius(self) -> tuple:
        return self.current_circle.radius

    def set_target_point(self, x, y):
        assert self.current_circle.is_contained(x, y)
        self.target_point = (x, y)

    def _get_vector(self):
        target_x, target_y = self.target_point
        current_center_x, current_center_y = self.current_circle.get_center()
        dx = target_x - current_center_x
        dy = target_y - current_center_y
        return dx, dy
    
    def _get_shrink_radius_per_frame(self, target_shrink_ratio, shrink_time, fps, shrink_mode="linear"):
        shrink_radius = (self.init_shrink_ratio - target_shrink_ratio) * self.init_circle.get_radius()
        if shrink_mode == "linear":
            return shrink_radius / (shrink_time * fps)
        
    def _shrink_per_frame(self, dx_per_frame, dy_per_frame, shrink_radius_per_frame):
        current_center_x, current_center_y = self.current_circle.get_center()
        new_center_x = current_center_x - dx_per_frame
        new_center_y = current_center_y - dy_per_frame

        current_radius = self.current_circle.get_radius()
        new_radius = current_radius - shrink_radius_per_frame

        self.current_circle = Circle(new_center_x, new_center_y, new_radius)
        return Circle(new_center_x, new_center_y, new_radius)
        

    def generate_target_point_randomly(self, distribution="uniform", random_seed=0):
        """
        Generate a point randomly inside the circle.
        """
        x, y = self.current_circle.generate_random_point_in_circle(distribution=distribution, random_seed=random_seed)
        return x, y

    def shrink(self, target_shrink_ratio: float = 0, shrink_time: float = 3, fps: float = 10, shrink_mode: str = "linear"):
        dx, dy = self._get_vector()
        if shrink_mode == "linear":
            dx_per_frame = dx / (shrink_time * fps) 
            dy_per_frame = dy / (shrink_time * fps)
        shrink_radius_per_frame = self._get_shrink_radius_per_frame(target_shrink_ratio, shrink_time, fps, shrink_mode="linear")
        
        shrink_circles = []
        for _ in range(shrink_time*fps):
            circle = self._shrink_per_frame(dx_per_frame, dy_per_frame, shrink_radius_per_frame)
            shrink_circles.append(circle)
        return shrink_circles

    def plot(self, shrink_circles, show_center: bool = True):
        fig = plt.figure()
        ax = plt.axes()

        for cicle in shrink_circles:
            c = patches.Circle(xy=(cicle.x, cicle.y), radius=cicle.radius, fill=False)
            ax.add_patch(c)
        plt.axis('scaled')
        ax.set_aspect('equal')
        plt.show()
            

    def animate(self, shrink_circles, show_center: bool = True):
        fig = plt.figure()
        
        ims = []

        for cicle in shrink_circles:
            ax = plt.axes()
            c = patches.Circle(xy=(cicle.x, cicle.y), radius=cicle.radius, fill=False)
            ax.add_patch(c)
            ax.set_aspect('equal')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ims.append([ax])

        ani = animation.ArtistAnimation(fig, ims, interval=100)
        plt.show()

if __name__ == "__main__":
    sc = ShrinkCircle()
    cs = sc.shrink()
    #sc.plot(cs)
    sc.animate(cs)