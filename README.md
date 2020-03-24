# an AI that plays Nintendo Switch games

Games to learn:
* Tetris 99 (current)
* Mario Maker 2

## How does it work?

Actor critic algorithms have done some pretty impressive stuff in the past few years. Actor critic algorithms such as [A2C](https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f) (Advantage Actor Critic) and A3C (Asychronous Advantage Actor Critic) use multiple neural networks to accomplish a single goal. In general, the actor network is making moves and the critic network is assigning rewards to those moves. The model I am putting together to play Tetris is a tweaked version of A2C, which will utilize a custom shaped reward function.

## To do list
* Clean discriminator database
* Create more states for environment (clearing lines, tetris, end game)

[Repo for sending commands to switch](https://github.com/wchill/SwitchInputEmulator)

Hardware Used:
* [Arduino Uno R3](https://www.amazon.com/Sintron-UNO-R3-ATMEGA328P-Arduino/dp/B073DYD97C/ref=sxin_2_ac_d_pm?ac_md=1-0-VW5kZXIgJDEw-ac_d_pm&cv_ct_cx=arduino+uno+r3&keywords=arduino+uno+r3&pd_rd_i=B073DYD97C&pd_rd_r=9305d6a8-c762-46e8-b3f8-356745bc8a6d&pd_rd_w=z4ysu&pd_rd_wg=MwvP1&pf_rd_p=0e223c60-bcf8-4663-98f3-da892fbd4372&pf_rd_r=RAANRF34YJKT950W9MVQ&psc=1&qid=1584519878&s=electronics)
* [USB to serial](https://www.amazon.com/gp/product/B01CYBHM26/ref=ppx_yo_dt_b_asin_title_o00_s00?ie=UTF8&psc=1)
* [Camlink 4k](https://www.amazon.com/gp/product/B07K3FN5MR/ref=ppx_yo_dt_b_asin_title_o02_s00?ie=UTF8&psc=1)