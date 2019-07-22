**WARNING: This is work-in-progress.**

## Prerequisites

`ArcadeLearningEnvironment` depends on the native ALE
library and on [GLFW](https://www.glfw.org/) which is used
for rendering. The native ALE library can be installed by
executing the following commands in a temporary working
directory:

```bash
git clone git@github.com:mgbellemare/Arcade-Learning-Environment.git
cd Arcade-Learning-Environment
mkdir build && cd build
cmake ..
make -j8
make install
```

GLFW can be installed by executing the following commands:

```bash
# For MacOS:
brew install --HEAD git glfw3

# For Linux:
sudo apt install libglfw3-dev libglfw3
```

After having installed these libraries you should be able
to use this library.

**NOTE:** The Swift Package Manager uses `pkg-config` to 
locate the installed libraries and so you need to make sure
that `pkg-config` is configured correctly. That may require
you to set the `PKG_CONFIG_PATH` environment variable
correctly.

## Supported Games

`ArcadeLearningEnvironment` will attempt to download the
game ROMs, if not found in the path you provide when
building the emulator. This means you never have to worry
about finding and downloading game ROMs yourself. The
following games are currently supported: Adventure,
Air Raid, Alien, Amidar, Assault, Asterix, Asteroids,
Atlantis, Bank Heist, Battle Zone, Beam Rider, Berzerk,
Bowling, Boxing, Breakout, Carnival, Centipede, Chopper
Command, Crazy Climber, Defender, Demon Attack, Double
Dunk, Elevator Action, Enduro, Fishing Derby, Freeway,
Frostbite, Gopher, Gravitar, Hero, Ice Hockey, James Bond,
Journey Escape, Kaboom, Kangaroo, Krull, Kung Fu Master,
Montezuma Revenge, Ms Pacman, Name This Game, Phoenix,
Pitfall, Pong, Pooyan, Private Eye, Q*bert, River Raid,
Road Runner, Robotank, Seaquest, Skiing, Solaris, Space
Invaders, Star Gunner, Tennis, Time Pilot, Tutankham,
Up N' Down, Venture, Video Pinball, Wizard Of Wor, Yars'
Revenge, Zaxxon.
