## Reinforcement Learning on the Nintendo Switch

Games to learn:
* Tetris 99 (current)
* Mario Maker 2

### How does it work?

Deep Q networks have set some impressive milestones for playing games in the atari domain. [Check here](https://towardsdatascience.com/advanced-dqns-playing-pac-man-with-deep-reinforcement-learning-3ffbd99e0814) to learn more about how they work and the optimizations necessary to get good performance.

### To do list
* Train on in-game data
* Create more states for environment (single line vs double vs triple vs tetris)
* Create optimization for skipping pre-game count down

### Dependencies

I recommend creating a virtual envioronment with python 3.5-3.6 as keras 2.1.5 with tensorflow 1.14 is used
Then run  
* pip install -r requirements.txt  
Currently only support for Windows machines

### References

[Repo for sending commands to switch](https://github.com/wchill/SwitchInputEmulator)  
[Repo for DQN](https://github.com/jakegrigsby/AdvancedPacmanDQNs)

### Hardware Used
* [Arduino Uno R3](https://www.amazon.com/Sintron-UNO-R3-ATMEGA328P-Arduino/dp/B073DYD97C/ref=sxin_2_ac_d_pm?ac_md=1-0-VW5kZXIgJDEw-ac_d_pm&cv_ct_cx=arduino+uno+r3&keywords=arduino+uno+r3&pd_rd_i=B073DYD97C&pd_rd_r=9305d6a8-c762-46e8-b3f8-356745bc8a6d&pd_rd_w=z4ysu&pd_rd_wg=MwvP1&pf_rd_p=0e223c60-bcf8-4663-98f3-da892fbd4372&pf_rd_r=RAANRF34YJKT950W9MVQ&psc=1&qid=1584519878&s=electronics)
* [USB to serial](https://www.amazon.com/gp/product/B01CYBHM26/ref=ppx_yo_dt_b_asin_title_o00_s00?ie=UTF8&psc=1)
* [Camlink 4k](https://www.amazon.com/gp/product/B07K3FN5MR/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&psc=1)