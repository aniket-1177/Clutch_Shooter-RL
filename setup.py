from setuptools import setup, find_packages

setup(
    name="clutch-shooter-rl",
    version="1.0.0",
    description="Basketball reinforcement learning: from Q-Tables to Deep RL with PPO",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pillow>=10.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "jupyter>=1.0.0"],
        "viz": ["tensorboard>=2.13.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
