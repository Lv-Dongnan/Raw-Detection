<template>
  <div class="home-page">
    <!-- 导航栏组件 -->
    <NavBar :modelChose="modelChose" 
    @update-modelChose="updateModelChose"
    @open-file-dialog="openFileDialog" 
    @start-ligament-segmentation="startLigamentSegmentation"
    @start-side-ligament-segmentation="startSideLigamentSegmentation"
    @start-3d-model-display="start3DModelDisplay" />

    <!-- 主体部分容器 -->
    <div class="main-container">
      <LeftBar :images="imageList" :index="index" @update-images="updateImages" />
      <MainContent :images="imageList" @update-images="updateImages" />
    </div>

    <!-- 导入图片的弹窗 -->
    <ImgSelect 
      :showDialog="showImportDialog" 
      @import-images="importImages" 
      @close-dialog="closeDialog" 
    />

    <!-- 显示加载动画 -->
    <div v-if="loading" class="loading-overlay">
      <span>加载中...</span>
    </div>

    <!-- 处理后的图片展示 -->
    <ImgOutput 
      v-if="imgResult.length > 0" 
      :images="imgResult" 
      @close-image-viewer="closeImageViewer"
    />

    <!-- 3D模型展示弹窗 -->
    <div v-if="showModelDisplay"   class="model-display-overlay">
      <model-display @request-model="handleRequestModel" ref="modelDisplay" @close="close3DModelDisplay" />
    </div>
  </div>
</template>

<script>
import axios from 'axios';
import NavBar from './homeBar/NavBar.vue';
import LeftBar from './homeBar/LeftBar.vue';
import MainContent from './homeBar/MainContent.vue';
import ImgSelect from './utils/imgInput.vue';
import ImgOutput from './utils/imgOutput.vue'; 
import ModelDisplay from './utils/modelDisplay.vue'; // 引入 ModelDisplay 组件

export default {
  name: 'HomePage',
  components: {
    NavBar,
    LeftBar,
    MainContent,
    ImgSelect,
    ImgOutput,
    ModelDisplay, // 注册 ModelDisplay 组件
  },
  data() {
    return {
      imageList: [], // 存储上传的图片文件列表
      showImportDialog: false, // 控制弹窗显示
      loading: false, // 控制加载动画
      imgResult: [], // 存储处理后的图片结果
      showModelDisplay: false, // 控制3D模型展示是否显示
      index:0,
      modelChose: 0,
    };
  },
  methods: {
    start3DModelDisplay() {
      this.loading = true;
      setTimeout(() => {
        this.showModelDisplay = true;  
        this.loading = false;
      }, 2000);  
    },

    // 打开导入图片的对话框
    openFileDialog() {
      this.imageList=[];
      this.showImportDialog = true;
    },

    // 导入图片
    importImages(files) {
      this.imageList = [...this.imageList, ...files];
      this.closeDialog();
    },

    closeDialog() {
      this.showImportDialog = false;
    },

    // 发送图片进行处理
    startLigamentSegmentation() {
      this.index=1;
      this.loading = true;
      const formData = new FormData();
      this.imageList.forEach((image) => {
        formData.append('images', image);
      });

      axios.post('http://	101.43.255.161:11625/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      .then(response => {
        this.imgResult = response.data.images;
        this.loading = false;
      })
      .catch(error => {
        console.error('图片上传失败', error);
        this.loading = false;
      });
    },

    startSideLigamentSegmentation() {
      this.index=2;
      this.loading = true;
      const formData = new FormData();
      this.imageList.forEach((image) => {
        formData.append('images', image);
      });

      axios.post('http://	101.43.255.161:11627/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      .then(response => {
        this.imgResult = response.data.images;
        this.loading = false;
      })
      .catch(error => {
        console.error('图片上传失败', error);
        this.loading = false;
      });
    },

    // 更新图片列表
    updateImages(updatedImages) {
      this.imageList = updatedImages;
    },

    // 关闭图片查看器
    closeImageViewer() {
      this.imgResult = [];
    },

    // 关闭3D模型展示
    close3DModelDisplay() {
      this.showModelDisplay = false; // 隐藏3D模型展示
    },
    updateModelChose(newValue) {
      this.modelChose = newValue; // 接收到子组件传来的新值后，更新父组件的 modelChose 数据
    },
    handleRequestModel() {

      // 直接访问子组件，通过 $refs 来传递数据
      this.$refs.modelDisplay.receiveModelChose(this.modelChose);

    },
  },
};
</script>

<style scoped>
.home-page {
  width: 100%;
  height: 100%;
}

.main-container {
  display: flex;
  justify-content: flex-start;
  align-items: flex-start;
}

.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  color: white;
  font-size: 20px;
  z-index: 1000;
}

img {
  max-width: 100%;
  max-height: 100%;
}

img {
  display: block;
  margin: 0 auto;
  border-radius: 5px;
}

/* 3D模型展示的蒙版与样式 */
.model-display-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}
</style>
