# WeaveWave: Towards Multimodal Music Generation

<div align="center">
   <img src="assets/logo/WeaveWave.png" alt="DSOTM" width="500px">
</div>
<p align="center">
   <i> WeaveWave: Towards Multimodal Music Generation </i>
</p>

## Overview

Artificial Intelligence Generated Content (AIGC), as a new generation of content production method, is reshaping the possibilities in the field of artistic creation. This project focuses on the vertical domain of music generation, exploring music generation models under multimodal conditions (text, images, videos).

**For humans**, music creation can be abstracted into two stages: **inspiration** and **implementation**. The former originates from the **fusion of diverse sensory experiences**: the stimulation of visual scenes, the resonance of literary imagery, the association of auditory fragments, and other cross-modal perceptions. The latter is manifested as the process of concretizing inspiration through singing, instrumental performance, etc.

**For machines**, can artificial intelligence music creation mimic these two stages? We believe that the task of **multimodal music generation** is precisely a simulation of "inspiration" and "implementation," where "inspiration" can be seen as multimodal data, and "implementation" as a music generation model.

<div align="center">
   <img src="assets/media/inspiration.png" alt="frame-1" width="500px">
</div>
<p align="center">
   <i> Music creation: humans and machines </i>
</p>

However, research on multimodal music generation has not yet garnered widespread attention, with most existing work confined to music understanding and generation within a single modality. This limitation clearly fails to fully capture the complex multimodal sources of inspiration in music creation.

To address this gap, we have adopted a **text-bridging** strategy, leveraging the potential of existing multimodal large language models and text-to-music generation models. This approach has led to the development of WeaveWave, a music generation framework that integrates multimodal inputs.

<div align="center">
   <img src="assets/media/frame-1.png" alt="frame-1" width="500px">
</div>
<p align="center">
   <i> WeaveWave: Text-Briding </i>
</p>

<!-- <br>

<div align="center">
   <img src="assets/media/frame-2.png" alt="frame-2" width="500px">
</div>
<p align="center">
   <i> WeaveWave: AudioLDM 2 </i>
</p>

<br>

<div align="center">
<img src="assets/media/frame-3.png" alt="frame-3" width="500px">
</div>
<p align="center">
   <i> WeaveWave: Based on MusicGen </i>
</p>

<br> -->

<!-- **autoregressive**
- MusicLM
- MusicGen
- ...

**non-autoregressive(LDM)**
- Stable Audio Open
- AudioLDM
- ...

<cite>[Audio Conditioning for Music Generation via Discrete Bottleneck Features](http://arxiv.org/abs/2407.12563)</cite>

### 2 directions:

understanding

generation -->

## Demo

<div align="center">
   <a href="assets/media/demo.mp4">
      <img src="assets/media/demo.png" alt="Demo" width="500px">
   </a>
</div>
<p align="center">
  <i> WeaveWave: Web app built with Gradio </i>
</p>

<!-- ### Review of Text-bridging multimodal music generation

其实，未尝不是一种 "end-to-end"

自然语言的力量，还没有被完全挖掘 -->
