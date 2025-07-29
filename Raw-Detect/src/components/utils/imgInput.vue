<template>
  <el-dialog
    :visible="showDialog"
    title="图片上传"
    width="80%"
    :before-close="cancelImport"
    :close-on-click-modal="false"
    :close-on-press-escape="false"
    class="img-input-dialog"
    :show-close="cancelImport"
  >
    <!-- 选择多张图片 -->
    <input 
      type="file" 
      ref="fileInput" 
      @change="handleFileChange" 
      accept="image/*" 
      multiple
      style="display: none"
    />
    <el-button @click="triggerFileInput" type="primary">选择图片</el-button>

    <!-- 图片预览 -->
    <div v-if="previewImages.length" class="preview-images">
      <div v-for="(image, index) in previewImages" :key="index" class="preview-item">
        <img :src="image.src" :alt="image.alt" class="preview-image" />
        <p>{{ image.alt }}</p>
      </div>
    </div>
    <!-- 没有图片时的空预览区域 -->
    <div v-else class="preview-images-empty">
      <p>没有选择图片</p>
    </div>

    <!-- 弹窗底部按钮 -->
    <span slot="footer" class="dialog-footer">
      <el-button @click="deleteImage">清除</el-button>
      <el-button type="primary" @click="importImages">确定</el-button>
    </span>
  </el-dialog>
</template>

<script>
export default {
  name: 'ImgInput',
  props: {
    showDialog: Boolean // 控制弹窗是否显示
  },
  data() {
    return {
      previewImages: []  // 用于存储预览的图片信息
    };
  },
  methods: {
    // 弹出文件选择框
    triggerFileInput() {
      this.$refs.fileInput.click();
    },

    // 处理选择的文件
    handleFileChange(event) {
      const files = event.target.files;
      this.previewImages = []; // 清空当前预览

      Array.from(files).forEach(file => {
        // 创建预览图像对象
        const reader = new FileReader();
        reader.onload = () => {
          this.previewImages.push({
            src: reader.result,  // 图片预览的base64数据
            alt: file.name,      // 文件名作为图片的描述
            file: file           // 文件本身，保留给父组件
          });
        };
        reader.readAsDataURL(file); // 读取文件并生成预览
      });
    },

    // 传递图片到父组件
    importImages() {
      // 获取所有选中的文件并传递给父组件
      
      const filesToSend = this.previewImages.map(image => image.file);
      this.$emit('import-images', filesToSend); // 向父组件发送选中的文件
      this.cancelImport();  // 关闭弹窗
    },

    //取消操作，关闭弹窗
    cancelImport() {
      const filesToSend =[];
      this.$emit('import-images', filesToSend); // 向父组件发送选中的文件
      
      this.previewImages = []; // 清空预览
    },

    deleteImage(){
      this.previewImages = []; // 清空预览
    }
  }
};
</script>

<style scoped>
.preview-images {
  display: flex;
  flex-wrap: wrap;
}

.preview-item {
  margin: 10px;
  text-align: center;
}

.preview-image {
  max-width: 100px;
  max-height: 100px;
}

.dialog-footer {
  text-align: right;
}
</style>
