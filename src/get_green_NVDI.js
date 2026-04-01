// ============================
// 1. 加载芝加哥网格
// ============================

var grid = ee.FeatureCollection(
  "projects/gen-lang-client-0107978268/assets/chicago_area"
);

Map.centerObject(grid, 10);


// ============================
// 2. Sentinel-2 云掩膜函数
// ============================

function maskS2clouds(image) {
  
  var qa = image.select('QA60');

  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask);
}


// ============================
// 3. 加载 Sentinel-2
// ============================

var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(grid)
  .filterDate('2021-01-01', '2025-01-01')
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
// 5. 时间统计
// ============================

var ndvi_mean = s2_ndvi
  .select('NDVI')
  .mean();

var ndvi_std = s2_ndvi
  .select('NDVI')
  .reduce(ee.Reducer.stdDev());


// ============================
// 6. 合并 mean 和 std
// ============================

var ndvi_combined = ndvi_mean.addBands(ndvi_std);


// ============================
// 7. 按 grid 计算 zonal stats
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