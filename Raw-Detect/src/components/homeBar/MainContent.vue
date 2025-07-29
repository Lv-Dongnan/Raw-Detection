<template>
  <div class="main-content">
    <h3> </h3>
    <div class="image-gallery" :style="{ gridTemplateColumns: `repeat(${columns}, 1fr)` }">
      <div v-for="(image, index) in imagePreviews" :key="index" class="image-item">
        <img :src="image.src" :alt="image.alt" class="image" />
        <p class="image-name">{{ image.alt }}</p>
      </div>
    </div>
    <div v-if="imagePreviews.length === 0" class="empty-gallery">
      <p>请选择分割图片</p>
    </div>
  </div>
</template>

<script>
export default {
  name: 'MainContent',
  props: {
    images: Array // 从父组件传递的图片列表
  },
  computed: {
    // 动态计算每行显示的列数
    columns() {
      return 8;
    },
    // 处理图片，生成预览图
    imagePreviews() {
      return this.images.map((file, index) => {
        return {
          src: URL.createObjectURL(file),  // 使用URL.createObjectURL生成图片的临时URL
          alt: file.name || `Image ${index + 1}`  // 如果文件有名字就用文件名，否则用“Image n”
        };
      });
    }
  }
};
</script>

<style scoped>
.main-content {
  flex-grow: 1;
  padding: 20px;
  display: flex;
  flex-direction: column; /* 垂直布局 */
  justify-content: center; /* 确保内容居中 */
  align-items: center; /* 水平居中 */
  min-height: calc(100vh - 160px); /* 最小高度占满剩余空间 */
}

.image-gallery {
  display: grid;
  gap: 20px;
  width: 100%; /* 确保宽度充满整个容器 */
}

.empty-gallery {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 80%; /* 填充整个高度 */
  text-align: center;
  font-size: 1.2em;
  color: #777;
}

.image-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.image {
  max-width: 100%;
  max-height: 150px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  object-fit: cover;
}

.image-name {
  margin-top: 10px;
  text-align: center;
  color: #555;
}
</style>
