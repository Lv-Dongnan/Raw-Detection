<template>
  <div class="nav-bar">
    <!-- 左侧部分，45% 宽度 -->
    <el-row class="left-section">
      <el-button @click="openFileDialog" type="primary">导入图片文件</el-button>
      <el-button @click="startLigamentSegmentation" type="success">模型推理</el-button>
      <el-button @click="startSideLigamentSegmentation" type="info">物体检测</el-button>
      <el-button @click="start3DModelDisplay" type="warning">检测结果展示</el-button>
    </el-row>

    <!-- 中间部分，40% 宽度 -->
    <el-row class="middle-section" justify="center" align="middle">
      <el-card class="console-output" shadow="always">
        <div class="console-messages">
          <el-alert v-for="(message, index) in consoleMessages" :key="index" :title="message" type="info" closable></el-alert>
        </div>
      </el-card>
    </el-row>

    <!-- 右侧部分，10% 宽度 -->
    <div class="right-section">
      <span class="art-text">RawDetect</span>
    </div>
  </div>
</template>

<script>
import { Button, Row, Card, Alert } from 'element-ui';

export default {
  name: 'NavBar',
  components: {
    ElButton: Button,
    ElRow: Row,
    ElCard: Card,
    ElAlert: Alert
  },
  props: {
    modelChose: {
      type: Number,
      required: true
    }
  },
  data() {
    return {
      consoleMessages: ['console: successfully generated.']  // 初始控制台消息
    };
  },
  methods: {
    openFileDialog() {
      this.addConsoleMessage('console: 导入图片文件');
      this.$emit('open-file-dialog');
    },
    startLigamentSegmentation() {
      this.addConsoleMessage('console: 模型推理...');
      this.$emit('update-modelChose', 0);
      this.$emit('start-ligament-segmentation');
    },
    startSideLigamentSegmentation() {
      this.addConsoleMessage('console: 开始物体检测');
      this.$emit('update-modelChose', 1);
      this.$emit('start-side-ligament-segmentation');
    },
    start3DModelDisplay() {
      this.addConsoleMessage('console: 检测结果展示');
      this.$emit('start-3d-model-display');
    },
    addConsoleMessage(message) {
      this.consoleMessages.unshift(message);  // 将新消息添加到控制台消息列表的开头
    }
  }
};
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Kaushan+Script&display=swap');

.nav-bar {
  display: flex;
  height: 120px;
  /* background-color: #f5f7fa; 更柔和的背景颜色 */
  background-image: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  justify-content: space-between;
  align-items: center;
  padding: 10px 20px;
  box-sizing: border-box;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); /* 增加更大且更柔和的阴影 */

}

.left-section {
  display: flex;
  justify-content: space-between;
  width: 45%;
  align-items: center;
}

.el-button {
  margin-right: 10px;
}

.middle-section {
  width: 40%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.console-output {
  width: 100%;
  max-height: 80px;
  overflow-y: auto; /* 启用垂直滚动条 */
}

.console-messages {
  max-height: 100%;
  overflow-y: auto;
  padding: 0 10px;
}

.right-section {
  width: 10%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.art-text {
  font-family: 'Kaushan Script', cursive;
  font-size: 30px;
  color: #333;
}
</style>
