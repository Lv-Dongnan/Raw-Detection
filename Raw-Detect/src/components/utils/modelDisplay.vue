<template>
  <div class="model-display-container">
    <div class="close-btn" @click="closeModelDisplay">❌</div>
    <div id="model-container"></div>
  </div>
</template>

<script>
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

export default {
  name: 'ModelDisplay',
  
  data() {
    return {
      scene: null,
      camera: null,
      renderer: null,
      model: null,
      controls: null,
      autoRotateSpeed: 0.01, // 自动旋转速度

      // 新增：鼠标拖动平移用到的变量
      isDragging: false,
      previousMousePosition: { x: 0, y: 0 },

      modelList:['https://raw.githubusercontent.com/LYU-DONGNAN/flex-model/main/model-CL.obj',
      'https://raw.githubusercontent.com/LYU-DONGNAN/flex-model/main/merged-model.obj'],

      modelChose:1,
    };
  },
  
  mounted() {
    
    this.$emit('request-model');

    setTimeout(() => {
      this.init3DModelDisplay();

    }, 100);

  },

  methods: {
    receiveModelChose(data) {
      this.modelChose=data;
    },
    init3DModelDisplay() {
      const container = this.$el.querySelector('#model-container');

      // 创建场景、相机和渲染器
      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
      this.renderer = new THREE.WebGLRenderer({ antialias: true });
      this.renderer.setSize(container.clientWidth, container.clientHeight);
      container.appendChild(this.renderer.domElement);

      // 添加光源
      const ambientLight = new THREE.AmbientLight(0x404040);
      this.scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
      directionalLight.position.set(1, 1, 1).normalize();
      this.scene.add(directionalLight);

      // 设置相机位置
      this.camera.position.set(0, 0, 30);

      // 加载 .obj 文件
      const loader = new OBJLoader();
      loader.load(
        this.modelList[this.modelChose],
        (object) => {
          object.scale.set(6, 6, 6);
          this.model = object;
          this.scene.add(object);
          object.position.set(0, 0, -500);
        },
        (xhr) => {
          console.log((xhr.loaded / xhr.total) * 100 + '% loaded');
        },
        (error) => {
          console.error('An error happened while loading the .obj file', error);
        }
      );

      // 添加 OrbitControls 控制器
      this.controls = new OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true; // 允许惯性
      this.controls.dampingFactor = 0.05;
      this.controls.screenSpacePanning = false;
      this.controls.minDistance = 10;
      this.controls.maxDistance = 100;
      this.controls.enableRotate = false; // 禁用左键旋转
      this.controls.enableZoom = false;   // 禁用滚轮缩放（由我们控制）
      this.controls.enablePan = true;     // 中键或右键平移
      this.controls.zoomSpeed = 1.2;
      this.controls.panSpeed = 1.2;
      this.controls.autoRotate = false;

      // ⭐新增：监听鼠标事件以实现自定义交互
      const dom = this.renderer.domElement;
      dom.addEventListener('mousedown', this.onMouseDown);
      dom.addEventListener('mouseup', this.onMouseUp);
      dom.addEventListener('mousemove', this.onMouseMove);
      dom.addEventListener('wheel', this.onMouseWheel);

      // 启动动画
      this.animate();
    },

    // ⭐新增：鼠标左键拖动模型
    onMouseDown(event) {
      if (event.button === 0) {
        this.isDragging = true;
        this.previousMousePosition = { x: event.clientX, y: event.clientY };
      }
    },
    onMouseUp() {
      this.isDragging = false;
    },
    onMouseMove(event) {
      if (!this.isDragging || !this.model) return;

      const deltaX = (event.clientX - this.previousMousePosition.x) * 0.5;
      const deltaY = (event.clientY - this.previousMousePosition.y) * 0.5;

      this.model.position.x += deltaX;
      this.model.position.y -= deltaY;

      this.previousMousePosition = { x: event.clientX, y: event.clientY };
    },

    // ⭐新增：滚轮缩放模型
    onMouseWheel(event) {
      if (!this.model) return;

      const scaleFactor = 1 + (event.deltaY > 0 ? -0.1 : 0.1);
      this.model.scale.multiplyScalar(scaleFactor);
    },

    animate() {
      requestAnimationFrame(this.animate);

      if (this.model) {
        this.model.rotation.y += this.autoRotateSpeed; // 模型旋转
      }

      if (this.controls) {
        this.controls.update(); // OrbitControls 更新
      }

      this.renderer.render(this.scene, this.camera);
    },
    closeModelDisplay() {
      this.$emit('close');
    }
  },
  beforeDestroy() {
    if (this.renderer) {
      this.renderer.dispose();
    }
    const dom = this.renderer?.domElement;
    if (dom) {
      dom.removeEventListener('mousedown', this.onMouseDown);
      dom.removeEventListener('mouseup', this.onMouseUp);
      dom.removeEventListener('mousemove', this.onMouseMove);
      dom.removeEventListener('wheel', this.onMouseWheel);
    }
  }
};
</script>

<style scoped>
.model-display-container {
  position: relative;
  width: 120vmin;
  height: 80vmin;
  background-color: rgba(255, 255, 255, 0);
  padding: 20px;
  box-sizing: border-box;
  display: flex;
  justify-content: center;
  align-items: center;
}

.close-btn {
  position: absolute;
  top: 10px;
  right: 10px;
  font-size: 30px;
  cursor: pointer;
  color: #333;
  background: none;
  border: none;
}

#model-container {
  width: 100%;
  height: 100%;
  border: 1px solid #ccc;
}
</style>
