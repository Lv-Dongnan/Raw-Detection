<template>
  <div class="sidebar">
    <!-- 图片信息表头 -->
    <div class="table-header">
      <h3>图片信息</h3>
    </div>
    <!-- 第一个表格展示图片信息 -->
    <div class="image-table">
      <table v-if="images.length">
        <thead>
          <tr>
            <th>序号</th>
            <th>图片名称</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(image, index) in images" :key="index">
            <td>{{ index + 1 }}</td>
            <td>{{ image.name }}</td>
          </tr>
        </tbody>
      </table>
      <!-- 如果没有图片则显示提示 -->
      <div v-else class="no-images">
        <p>暂无图片</p>
      </div>
    </div>

    <!-- 模型信息表头 -->
    <div class="table-header">
      <h3>模型信息</h3>
    </div>
    <!-- 第二个表格展示模型信息 -->
    <div class="model-info">
      <table class="info-table">
        <tr>
          <td>模型选择：</td>
          <td><strong>{{ modelList[index].name }}</strong></td>
        </tr>
        <tr>
          <td>模型精度：</td>
          <td><strong>{{ modelList[index].accuracy }}</strong></td>
        </tr>
        <tr>
          <td>已选择图片：</td>
          <td><strong>{{ images.length }} 张</strong></td>
        </tr>
      </table>
    </div>
  </div>
</template>


<script>
export default {
  name: 'LeftBar',
  props: {
    images: Array,
    index: {
      type: Number,
      default: 0
    }
  },
  data() {
    return {
      // 模型列表，包含模型名称和精度
      modelList: [
        // { name: '未选择模型', accuracy: '未选择精度' },
        // { name: 'PL', accuracy: '0.9745' },
        // { name: 'CL', accuracy: '0.9547' }，
        { name: 'Lora-MMD', accuracy: '0.9547' },
        { name: 'Lora-MMD', accuracy: '0.9547' },
        { name: 'Lora-MMD', accuracy: '0.9547' },
        
      ]
    };
  },
  methods: {
    // 更新图片列表
    updateImages(newImages) {
      this.$emit('update-images', newImages);
    }
  }
};
</script>


<style scoped>
.sidebar {
  width: 250px;  /* 固定宽度 */
  /* background-color: #f4f4f4; */
  background-image: linear-gradient(to top,  #e7f0fd 0%,#accbee 100%);
  padding-top: 20px;
  border-right: 2px solid #ccc;
  display: flex;
  flex-direction: column;
  height: calc(100vh - 140px);  /* 设置高度为页面高度减去80px */
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); /* 增加更大且更柔和的阴影 */

}

h3 {
  margin-bottom: 0;
}

.table-header {
  /* background-color: #f1f1f1;  添加背景色 */
  padding: 3px 8px;  /* 减少padding，表头更加紧凑 */
  text-align: center;
  font-size: 16px;
  border-radius: 5px;
  margin-bottom: 3px;  /* 减少表头和表格的间距 */
}

.image-table {
  display: flex;
  flex-direction: column;
  overflow-y: auto;  /* 当内容超出时添加滚动条 */
  background-color: white;  /* 给表格区域添加白色背景 */
  height: 400px;  /* 固定表格区域的高度 */
  margin-top: 5px;  /* 增加与表头的间距 */
  padding: 10px;  /* 给表格区域添加10px的内边距 */
  border-radius: 5px;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 5px;  /* 缩短表头与表格之间的距离 */
}

th, td {
  padding: 8px;
  text-align: left;
  border: 1px solid #ddd;
}



.no-images {
  flex-grow: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;  /* 保证提示区域占据剩余空间 */
  /* color: #888; */
  text-align: center;
}

p {
  font-size: 16px;
  /* color: #888; */
}

/* 模型信息表格样式 */
.model-info {
  background-color: white;
  margin-top: 5px;  /* 增加两个表格之间的间距 */
  margin-bottom: 5px;  /* 减小模型信息和下面表格的间距 */
  padding: 10px;
  border-radius: 5px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.info-table {
  width: 100%;
  border-collapse: collapse;
}

.info-table td {
  padding: 8px;
  text-align: left;
}

.info-table td:first-child {
  font-weight: bold;
}


</style>
