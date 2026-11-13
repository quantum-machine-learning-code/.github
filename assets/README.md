# 组织图标素材 / Org mark assets

唯一源图是 [`quantum-mark.svg`](quantum-mark.svg)：`viewBox="0 0 100 100"`，线条用 `currentColor`，会跟随容器的文字颜色。改图标只需改这一个文件，再用 `render_assets.py` 重新生成 PNG。

| 文件 | 尺寸 | 背景 / 线条 | 用途 |
| --- | --- | --- | --- |
| `quantum-mark.svg` | 矢量 | 透明 / `currentColor` | 源图，网页与文档内联引用 |
| `quantum-mark.png` | 512×512 | 透明 (RGBA) / 深色线 | 浅色背景上叠加、幻灯片、排版 |
| `quantum-avatar.png` | 512×512 | 白底 / 深色线 | **推荐：GitHub 组织头像**（通用） |
| `quantum-avatar-dark.png` | 512×512 | 深底 `#0d1117` / 浅色线 | 深色主题、深色背景备选 |

三版 PNG 都由 [`render_assets.py`](render_assets.py) 从同一 SVG 用 headless Chrome / Edge 渲染，图标占画布 90%，四周留白 5%，适合 GitHub 头像的圆形裁切。透明版含 alpha 通道，深色线条只适合浅色背景；深色背景请用 `quantum-avatar-dark.png`。

**设置组织头像**：GitHub 没有公开的头像上传 API，需到组织设置页手动上传 PNG：
<https://github.com/organizations/quantum-machine-learning-code/settings/profile> → *Profile picture* → *Upload a new picture*，选择 `quantum-avatar.png`。

**复现**：`python render_assets.py`（仅用 Python 标准库 + 本机 Chrome / Edge headless，无第三方依赖）。
