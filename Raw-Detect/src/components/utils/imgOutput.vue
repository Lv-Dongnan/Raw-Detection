<template>
    <div class="image-viewer-overlay">
      <div class="image-viewer">
        <!-- 显示左箭头，切换图片 -->
        <button class="prev" @click="prevImage">＜</button>
        
        <!-- 显示当前图片 -->
        <div class="image-container">
          <img :src="images[currentIndex]" alt="Processed Image" />
        </div>
        
        <!-- 显示右箭头，切换图片 -->
        <button class="next" @click="nextImage">＞</button>
        
        <!-- 图片索引与总数 -->
        <div class="image-info">
          <span>{{ currentIndex + 1 }} / {{ images.length }}</span>
        </div>
  
        <!-- 关闭按钮 -->
        <button class="close" @click="closeViewer">❌</button>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    props: {
      images: Array, // 接收图片数组
    },
    data() {
      return {
        currentIndex: 0, // 当前显示的图片索引
      };
    },
    methods: {
      prevImage() {
        if (this.currentIndex > 0) {
          this.currentIndex--;
        }
      },
      nextImage() {
        if (this.currentIndex < this.images.length - 1) {
          this.currentIndex++;
        }
      },
      closeViewer() {
        this.$emit('close-image-viewer');
      },
    },
  };
  </script>
  
  <style scoped>
  .image-viewer-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.8); /* Dark background */
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
  }
  
  .image-viewer {
    position: relative;
    background-color: transparent;
    padding: 20px;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  
  .image-container {
    max-width: 80%; /* Reduced size of the image */
    max-height: 80%;
    overflow: hidden;
    position: relative;
  }
  
  .image-container img {
    width: 100%;
    height: auto;
    display: block;
    object-fit: contain;
  }
  
  .prev, .next {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    background-color: rgba(255, 255, 255, 0); /* Semi-transparent white */
    color: white;
    border: none;
    font-size: 30px;
    cursor: pointer;
    padding: 15px;
    border-radius: 50%;
    z-index: 2;
  }
  
  .prev {
    left: 10px;
  }
  
  .next {
    right: 10px;
  }
  
  .image-info {
    position: absolute;
    bottom: 20px;
    background-color: rgba(0, 0, 0, 0.5);
    color: white;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 14px;
  }
  
  .close {
    position: absolute;
    top: 10px;
    right: 10px;
    background-color: rgba(255, 255, 255, 0);
    color: black;
    border: none;
    font-size: 24px;
    cursor: pointer;
    padding: 10px;
    border-radius: 50%;
    z-index: 2;
  }
  </style>
  