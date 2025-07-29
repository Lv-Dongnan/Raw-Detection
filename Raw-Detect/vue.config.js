const { defineConfig } = require('@vue/cli-service');

module.exports = defineConfig({
  transpileDependencies: true,
  pluginOptions: {
    electronBuilder: {
      builderOptions: {
        win: {
          icon: 'src/assets/logo.ico', // 设置 Windows 平台的图标
        },
        mac: {
          icon: 'src/assets/logo.icns', // 设置 macOS 平台的图标
        },
        linux: {
          icon: 'src/assets/logo.png', // 设置 Linux 平台的图标
        }
      }
    }
  },
  chainWebpack: config => {
    // 配置 Webpack 以支持 .obj 文件
    config.module
      .rule('obj')
      .test(/\.obj$/)
      .type('asset/resource') // 使用 Webpack 5 的资源模块
      .end();
  }
});
