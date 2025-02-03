
# MiniGrid with Sprites

![License](https://img.shields.io/github/license/username/repository)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/figures/DoorKey.gif" alt="Door Key" height="250">
  <img src="/figures/Unlock.gif" alt="Unlock" height="250">
</div>
<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/figures/Unlock_1.gif" alt="Unlock" height="250">
  <img src="/figures/Crossing.gif" alt="Unlock" height="250">
</div>

> An alternate way to view the Minigrid environment using animations and sprites. Intended to be used as a teaching tool for kids to promote more engagment and interest in Renforcemnt Learning.

## Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Roadmap](#roadmap)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## About the Project

Designed to engage students in learning about AI and reinforcement learning specifically, Minigrid with Sprites adds an entirely new rendering manager to Minigrid. This rendering manager utilizes Pyglet along with tons of custom logic to create a beautifully rendered environment for any Minigrid environment.

## Features

- ✨ **Sprites and Animations:** Using pyglet Minigrid environmnets can be rendered with any sprite.
- ✨ **Seperate Rendering Logic:** The rendering logic is now seperated and much easier to customize.

## Getting Started

```bash
# Clone the repository
git clone https://github.com/NSChristopher/Minigrid_with_Sprites.git

# Navigate into the project directory
cd repository

# Install dependencies
pip install -r requirements.txt
```

## Usage

Import the PrettyRenderingManager along with the gym and your desired environment.

```python
# Example usage
import gymnasium as gym
from minigrid.envs import LavaGapEnv
from minigrid.rendering.pretty_rendering_manager import PrettyRenderingManager
```

Pass the new PrettyRenderingManager into the env.
```python
env = LavaGapEnv(render_mode='rgb_array', rendering_manager=PrettyRenderingManager,size=6)
```

## Contact

Noah Christopher - noah.dev@outlook.com

Project Link: [Minigrid_with_Sprites](https://github.com/NSChristopher/Minigrid_with_Sprites)

## Acknowledgements

- [MiniGrid](https://github.com/Farama-Foundation/Minigrid)
- [Tiny Dungeon by ORYX DESIGN LAB](https://www.oryxdesignlab.com/products/p/tiny-dungeon-tileset)
- [PUNY CHARACTERS by Shade](https://opengameart.org/content/puny-characters)
