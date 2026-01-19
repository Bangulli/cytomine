<template>
  <div>
    <div class="wsi-retrieval">
      <h1>{{ $t('wsi similarity search') }}</h1>

    </div>

    <div>
      <h5>{{ $t('Search Parameters') }}</h5>

      <b-field>
        <template #label>
          Staining (optional)
          <b-tooltip multilined label="Staining metadata filtration. Takes the codes or string values of stainings from SNOMED as input. You can combine multiple commands by AND gates with + or OR gates with |. A value can be excluded by adding a ! in front of the parameter. Example for returning only H&E or Hematoxylin & Antibody stained images: 36879007 AND 12710003 OR 12710003 AND Antibody">
            <b-icon pack="fas" icon="info-circle" />
          </b-tooltip>
        </template>

        <b-input
          v-model="staining"
          placeholder="Staining (coming soon)"
          disabled
        />
      </b-field>

      <b-field>
        <template #label>
          Organ (optional)
          <b-tooltip label="This feature is planned but not yet available.">
            <b-icon pack="fas" icon="info-circle" />
          </b-tooltip>
        </template>

        <b-input
          v-model="organ"
          placeholder="Organ (coming soon)"
          disabled
        />
      </b-field>

      <b-field>
        <template #label>
          Species (optional)
          <b-tooltip label="This feature is planned but not yet available.">
            <b-icon pack="fas" icon="info-circle" />
          </b-tooltip>
        </template>

        <b-input
          v-model="species"
          placeholder="Species (coming soon)"
          disabled
        />
      </b-field>

      <b-field>
        <template #label>
          Diagnosis
          <b-tooltip label="This feature is planned but not yet available.">
            <b-icon pack="fas" icon="info-circle" />
          </b-tooltip>
        </template>

        <b-input
          v-model="diagnosis"
          placeholder="Diagnosis (coming soon)"
          disabled
        />
      </b-field> 

      <!-- k -->
      <b-field label="k (number of results)">
        <b-input type="number" v-model.number="k"></b-input>
      </b-field>

      <!-- local search -->
      <b-field>
        <b-checkbox v-model="localSearch">
          Search in this project only
        </b-checkbox>
      </b-field>

      <button
        class="button is-small is-fullwidth"
        @click="searchSimilarImages"
        :disabled="isLoading"
        :class="{ 'is-loading': isLoading }"
      >
        {{ $t('Search') }}
      </button>

      <div class="results-container">
        <div
          v-if="similarityRows.length"
          class="results-table-container"
        >
          <table class="table is-fullwidth is-striped is-hoverable">
            <thead>
              <tr>
                <th>#</th>
                <th>Image ID</th>
                <th>Distance</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in similarityRows" :key="row.id">
                <td>{{ row.rank }}</td>
                <td>{{ row.name }}</td>
                <td>{{ row.distance.toFixed(4) }}</td>
                <td>
                  <button
                    class="button is-small"
                    @click="addImage(row.id)"
                  >
                    Display
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <p v-else class="has-text-grey is-italic no-results">
          No results yet.
        </p>
      </div>
    </div>
  </div>
</template>

<script>
import {Cytomine, ImageInstanceCollection} from '@/api';
import {get} from '@/utils/store-helpers';

export default {
  name: 'WsiCbirPanel',
  props: {
    index: String,
  },
  data() {
    return {
      data: null,
      isLoading: false, 

      staining: '',
      organ: '',
      species: '',
      diagnosis: '',
      localSearch: true,
      k: 3,
    };
  },
  computed: {
    sleep() {
      const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
      return sleep;
    },
    viewerWrapper() {
      return this.$store.getters['currentProject/currentViewer'];
    },
    project: get('currentProject/project'),
    image() {
      return this.viewerWrapper.images[this.index].imageInstance;
    },
    viewerModule() {
      return this.$store.getters['currentProject/currentViewerModule'];
    },
    // Build nice rows from the similarities, skipping index 0
    similarityRows() {
      if (!this.data || !Array.isArray(this.data.similarities)) {
        return [];
      }

      // similarities is of form: [ [ "IMAGE_id", distance ], ... ]
      // skip the first element (the query image itself)
      return this.data.similarities.slice(1).map(([id, distance, name], idx) => ({
        rank: idx + 1,
        name,
        distance,
        id,
      }));
    },
  },
  methods: {
    async searchSimilarImages() {
      if (this.isLoading) {
        return;
      }

      const body = {
        query: this.image.baseImage,
        datasets: '',
        staining: this.staining || '',
        organ: this.organ || '',
        species: this.species || '',
        diagnosis: this.diagnosis || '',
        projectId: this.localSearch ? this.project.id.toString() : '',
        k: (this.k ?? 3) + 1,
      };

      this.isLoading = true;
      this.data = null;

      try {
        // 1) submit job
        const submit = await Cytomine.instance.api.post('wsi-cbir/retrieval', body);
        const jobId = submit.data.jobId || submit.data.job_id;

        // 2) poll job
        const pollIntervalMs = 1000;
        const timeoutMs = 10 * 60 * 1000; // 10 minutes
        const start = Date.now();

        let done = false;
        while (!done) {
          if (Date.now() - start > timeoutMs) {
            throw new Error('Retrieval timed out (client-side).');
          }

          const s = await Cytomine.instance.api.get(`wsi-cbir/jobs/${jobId}`);
          const {state, result, error} = s.data;

          // optional UI updates
          // this.loadingMessage = message;
          // this.loadingProgress = progress;

          if (state === 'DONE') {
            this.data = result;
            done = true;
          } else if (state === 'FAILED') {
            throw new Error(error || 'Job failed');
          } else if (state === 'CANCELED') {
            throw new Error('Job canceled');
          } else {
            await this.sleep(pollIntervalMs); // <-- limits to ~1 request/sec
          }
        }
      } catch (e) {
        console.error('Error while searching similar images:', e);
      } finally {
        this.isLoading = false;
      }
    },
    async findImage(id) {
      let images = (await ImageInstanceCollection.fetchAll({
        filterKey: 'project',
        filterValue: this.project.id,
      })).array;
      return images.find(img => Number(img.baseImage) === Number(id)) || null;
    },
    async addImage(imageId) {
      console.log('trying to load another image');
      try {
        const imgToLoad = await this.findImage(imageId);
        await imgToLoad.fetch();
        let slice = await imgToLoad.fetchReferenceSlice();
        await this.$store.dispatch(this.viewerModule + 'addImage', {image: imgToLoad, slices: [slice]});
      } catch (error) {
        console.log(error);
        this.$notify({type: 'error', text: this.$t('notif-error-add-viewer-image')});
      }
    },
  },
};
</script>

<style scoped>
.wsi-retrieval {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.results-container {
  margin-top: 1rem;
}

.results-table-container {
  max-height: 300px;
  overflow-y: auto;
  margin-top: 0.5rem;
}

.no-results {
  margin-top: 1rem;
}

/* You can remove this if you no longer use the textarea */
.results-textfield {
  width: 100%;
  height: 300px;
  padding: 0.5rem;
  font-family: monospace;
  font-size: 0.85rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  resize: vertical;
  overflow: auto;
  background-color: #f5f5f5;
}
</style>
