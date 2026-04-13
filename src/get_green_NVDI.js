// ============================
// 1. 加载芝加哥网格
// ============================

var grid = ee.FeatureCollection(
  "projects/gen-lang-client-0107978268/assets/chicago_area"
);

Map.centerObject(grid, 10);


// ============================
// 2. Sentinel-2 云掩膜
// ============================

function maskS2clouds(image) {

  var scl = image.select('SCL');

  var mask = scl.neq(3)
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));

  return image.updateMask(mask);
}


// ============================
// 3. 加载 Sentinel-2
// ============================

var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(grid)
  .filterDate('2022-01-01', '2023-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds);


// ============================
// 4. 计算 NDVI
// ============================

var addNDVI = function(image) {
  
  var ndvi = image.normalizedDifference(['B8','B4'])
    .rename('NDVI');

  return image.addBands(ndvi);
};

var s2_ndvi = s2.map(addNDVI);


// ============================
// 5. 计算时间统计
// ============================

var ndvi_mean = s2_ndvi
  .select('NDVI')
  .mean()
  .rename('NDVI_mean');;

var ndvi_std = s2_ndvi
  .select('NDVI')
  .reduce(ee.Reducer.stdDev())
  .rename('NDVI_std');;


// ============================
// 6. 合并
// ============================

var ndvi_combined = ndvi_mean.addBands(ndvi_std);


// ============================
// 7. grid zonal stats
// ============================

var stats = ndvi_combined.reduceRegions({
  
  collection: grid,
  
  reducer: ee.Reducer.mean(),
  
  scale: 500
});


// ============================
// 8. 导出 CSV
// ============================

Export.table.toDrive({
  
  collection: stats,
  
  description: "Chicago_NDVI_grid",
  
  fileFormat: "CSV"
});


// ========== 夜间遥感灯光遥感数据 ==============
// ================================
// 1. 读取芝加哥网格
// ================================
var grid = ee.FeatureCollection(
  "projects/gen-lang-client-0107978268/assets/chicago_area"
);

// 可视化网格
Map.centerObject(grid, 10);
Map.addLayer(grid, {}, "Chicago Grid");


// ================================
// 2. 选择夜间灯光数据 (VIIRS)
// ================================
// NASA Black Marble 年度夜间灯光


var night = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
  .filterDate('2022-01-01', '2023-12-31')
  .select('avg_rad')
  .mean();


// ================================
// 3. 裁剪到芝加哥区域
// ================================
night = night.clip(grid);


// ================================
// 4. 可视化
// ================================
print('2022-2023 stats', night.reduceRegion({
  reducer: ee.Reducer.minMax(),
  geometry: grid,
  scale: 500,
  maxPixels: 1e13
}));
var vis = {
  min: 0,
  max: 320,
  palette: ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']
};

Map.addLayer(night, vis, 'Night Light 2022-2023');


// ================================
// 5. 导出为 GeoTIFF
// ================================

// 2022-2023
Export.image.toDrive({
  image: night,
  description: 'Chicago_NightLight_2022_2023',
  folder: 'GEE_Chicago',
  fileNamePrefix: 'Chicago_NTL_2022_2023',
  region: grid.geometry(),
  scale: 500,   // VIIRS 推荐 500m
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});