import Vue from 'vue'
import App from './App.vue'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'  // 引入 Element UI 样式
import axios from 'axios'
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';  // 使用正确的路径


// 将 THREE 和 OBJLoader 全局挂载到 Vue 原型上
Vue.prototype.$THREE = THREE;
Vue.prototype.$OBJLoader = OBJLoader;

Vue.config.productionTip = false

// 使用 Element UI
Vue.use(ElementUI)

// 配置全局 axios
Vue.prototype.$axios = axios

new Vue({
  render: h => h(App),
  mounted() {
    // 最大化窗口
    if (window.innerWidth < screen.width || window.innerHeight < screen.height) {
      window.moveTo(0, 0);
      window.resizeTo(screen.width, screen.height);
    }
  }
}).$mount('#app')
